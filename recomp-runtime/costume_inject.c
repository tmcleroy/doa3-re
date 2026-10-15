#include "costume_inject.h"

#include "d3d_draw_model.h"
#include "d3d_frame_adapter.h"
#include "runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* DOA3 keeps each fighter's 23 bone matrices (row-vector, 64 bytes each) in
   consecutive banks; a fighter part's WORLD0 is a verbatim copy of one.
   Measured in the attract demo: 0x004785D0 and 0x00478B90. */
enum { FIGHTERS = 2, DOA3_BONES = 23, DOAX_BONES = 32, MAX_WORLDS = 4 };
static const uint32_t bank_base[FIGHTERS] = {0x004785d0u, 0x00478b90u};

typedef struct Texture {
    RecompD3dTextureDesc desc;
    uint8_t *bytes, *palette;
    uint32_t byte_count, palette_count;
} Texture;

typedef struct Part {
    uint32_t stride, fvf, primitive, vertex_count, index_count, flags, world_count;
    int32_t texture;
    uint32_t bones[MAX_WORLDS];
    float offsets[MAX_WORLDS][16];
    uint8_t *vertices;
    uint16_t *indices;
    float *skinned; /* vertex_count * 8 floats: position, normal, uv */
    bool has_material;
    float material[17]; /* source D3DMATERIAL8: diffuse, ambient, specular, emissive, power */
    /* Soft part (hair, chest): spring stiffness, damping, gain, offset limit,
       and a per-vertex 0-1 jiggle mask learned from the source's own motion. */
    bool soft;
    float soft_params[4];
    float *mask;
} Part;

typedef struct Asset {
    uint32_t relative_count;
    uint32_t relative_bone[16], relative_parent[16];
    float relative[16][16];
    uint32_t hierarchy_count;
    uint32_t hierarchy_bone[32], hierarchy_parent[32];
    float hierarchy_offset[32][3];
    Texture *textures;
    uint32_t texture_count;
    Part *parts;
    uint32_t part_count;
} Asset;

static struct {
    bool configured, enabled;
    unsigned fighter_mask;
    Asset assets[FIGHTERS];
    Asset *fighter_asset[FIGHTERS];
    uint32_t injected_present[FIGHTERS];
    /* Lighting of the fighter's last lit draw: the draw that triggers the
       replacement may be an unlit part (eyes, effects). */
    bool have_lit[FIGHTERS];
    RecompD3dDirectionalLighting lit[FIGHTERS];
    float lit_material[FIGHTERS][17];
    /* Per fighter: soft-part spring state (offset, velocity) and the last two
       positions of each bone, for acceleration. */
    float *soft_state[FIGHTERS];
    float history[FIGHTERS][2][DOAX_BONES][3];
    uint32_t history_present[FIGHTERS], history_count[FIGHTERS];
    /* Vertex-data span of each fighter's own draws, by present: props from the
       same loaded model (shoe straps, ribbons) fall inside it. */
    uint32_t span_present[FIGHTERS], span_low[FIGHTERS], span_high[FIGHTERS];
    uint32_t last_low[FIGHTERS], last_high[FIGHTERS];
} inject;

/* DOAX bone-local -> DOA3 bone-local rotation, row vectors (v3 = vX * C).
   Torso, head, hands and feet share axes. Limb rows map the bone direction
   and the knee/elbow bend axis (cross of the two segment directions at the
   most bent sample) between the games; hands and feet were confirmed by
   matching the ankle/wrist relative rotations. */
static const int8_t correction[15][9] = {
    {1,0,0, 0,1,0, 0,0,1}, {1,0,0, 0,1,0, 0,0,1}, {1,0,0, 0,1,0, 0,0,1},
    {1,0,0, 0,1,0, 0,0,1},                      /* 3 left foot */
    {0,0,-1, 0,1,0, 1,0,0},                     /* 4 left forearm */
    {1,0,0, 0,1,0, 0,0,1},                      /* 5 left hand */
    {0,-1,0, 0,0,-1, 1,0,0},                    /* 6 left thigh */
    {0,-1,0, 0,0,1, -1,0,0},                    /* 7 left shin */
    {0,0,1, 0,1,0, -1,0,0},                     /* 8 left upper arm */
    {1,0,0, 0,1,0, 0,0,1},                      /* 9 right foot */
    {0,0,1, 0,1,0, -1,0,0},                     /* 10 right forearm */
    {1,0,0, 0,1,0, 0,0,1},                      /* 11 right hand */
    {0,1,0, 0,0,-1, -1,0,0},                    /* 12 right thigh */
    {0,1,0, 0,0,1, 1,0,0},                      /* 13 right shin */
    {0,0,-1, 0,1,0, 1,0,0},                     /* 14 right upper arm */
};

static void multiply(const float a[16], const float b[16], float out[16])
{
    float r[16];
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j) {
            double s = 0;
            for (unsigned k = 0; k < 4; ++k) s += (double)a[i*4+k] * b[k*4+j];
            r[i*4+j] = (float)s;
        }
    memcpy(out, r, sizeof r);
}

static void invert_affine(const float m[16], float out[16])
{
    /* Rotation may carry scale; invert the 3x3 generally. */
    const float *a = m;
    double det = a[0]*(a[5]*a[10]-a[6]*a[9]) - a[1]*(a[4]*a[10]-a[6]*a[8]) + a[2]*(a[4]*a[9]-a[5]*a[8]);
    double inv = det != 0 ? 1.0/det : 0;
    float r[16] = {0};
    r[0] = (float)((a[5]*a[10]-a[6]*a[9])*inv); r[1] = (float)((a[2]*a[9]-a[1]*a[10])*inv); r[2] = (float)((a[1]*a[6]-a[2]*a[5])*inv);
    r[4] = (float)((a[6]*a[8]-a[4]*a[10])*inv); r[5] = (float)((a[0]*a[10]-a[2]*a[8])*inv); r[6] = (float)((a[2]*a[4]-a[0]*a[6])*inv);
    r[8] = (float)((a[4]*a[9]-a[5]*a[8])*inv); r[9] = (float)((a[1]*a[8]-a[0]*a[9])*inv); r[10] = (float)((a[0]*a[5]-a[1]*a[4])*inv);
    for (unsigned c = 0; c < 3; ++c)
        r[12+c] = -(m[12]*r[c] + m[13]*r[4+c] + m[14]*r[8+c]);
    r[15] = 1;
    memcpy(out, r, sizeof r);
}

/* Mirrors animation_skeleton.c swing(): the rotation taking row a to row b. */
static void swing(const float a[3], const float b[3], float m[16])
{
    memset(m, 0, 64); m[15] = 1;
    double c = (double)a[0]*b[0] + (double)a[1]*b[1] + (double)a[2]*b[2];
    float axis[3] = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    if (1.0 + c <= 1e-6) {
        unsigned least = 0;
        for (unsigned i = 1; i < 3; ++i) if (fabsf(a[i]) < fabsf(a[least])) least = i;
        float basis[3] = {0}; basis[least] = 1;
        float x[3] = {a[1]*basis[2]-a[2]*basis[1], a[2]*basis[0]-a[0]*basis[2], a[0]*basis[1]-a[1]*basis[0]};
        float l = sqrtf(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        for (unsigned i = 0; i < 3; ++i) x[i] /= l;
        for (unsigned r = 0; r < 3; ++r)
            for (unsigned k = 0; k < 3; ++k) m[r*4+k] = 2.0f*x[r]*x[k] - (r == k);
        return;
    }
    for (unsigned r = 0; r < 3; ++r)
        for (unsigned k = 0; k < 3; ++k)
            m[r*4+k] = (float)((r == k ? c : 0.0) + (double)axis[r]*axis[k]/(1.0 + c));
    for (unsigned i = 0; i < 3; ++i) {
        unsigned u = (i+1)%3, v = (i+2)%3;
        m[u*4+v] += axis[i]; m[v*4+u] -= axis[i];
    }
}

/* Mirrors animation_skeleton.c blend(). */
static void blend(const float a[16], const float b[16], float w, float out[16])
{
    float at[16] = {0}, rel[16], rot[16] = {0};
    for (unsigned r = 0; r < 3; ++r) for (unsigned c = 0; c < 3; ++c) at[r*4+c] = a[c*4+r];
    at[15] = 1;
    multiply(b, at, rel);
    float axis[3] = {rel[6]-rel[9], rel[8]-rel[2], rel[1]-rel[4]};
    float len = sqrtf(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
    if (len > 0) for (unsigned i = 0; i < 3; ++i) axis[i] /= len;
    float cosine = (rel[10]+rel[5]+rel[0]-1)*0.5f;
    double angle = atan2((double)len*0.5, cosine)*w;
    float s = (float)sin(angle), c = (float)cos(angle);
    for (unsigned r = 0; r < 3; ++r)
        for (unsigned k = 0; k < 3; ++k) rot[r*4+k] = (r == k ? c : 0) + (1-c)*axis[r]*axis[k];
    for (unsigned i = 0; i < 3; ++i) {
        unsigned u = (i+1)%3, v = (i+2)%3;
        rot[u*4+v] += s*axis[i]; rot[v*4+u] -= s*axis[i];
    }
    rot[15] = 1;
    float result[16];
    multiply(rot, a, result);
    for (unsigned i = 12; i < 15; ++i) result[i] = (1-w)*a[i] + w*b[i];
    result[15] = 1;
    memcpy(out, result, sizeof result);
}

/* DOAX's 32-bone bank posed by a DOA3 skeleton (see derived_joints). */
static void doax_bank(const Asset *a, const float doa3[DOA3_BONES][16], float b[DOAX_BONES][16])
{
    memset(b, 0, sizeof(float) * DOAX_BONES * 16);
    for (unsigned i = 0; i < 15; ++i) {
        float c[16] = {0};
        for (unsigned r = 0; r < 3; ++r) for (unsigned k = 0; k < 3; ++k) c[r*4+k] = correction[i][r*3+k];
        c[15] = 1;
        multiply(c, doa3[i], b[i]);
    }
    /* Keep DOA3's rotations but rebuild positions from the source's own bone
       offsets, parents first, so joint skins built for the source proportions
       still close. DOAX-only bones take their whole source-frame transform. */
    for (uint32_t h = 0; h < a->hierarchy_count; ++h) {
        const uint32_t bone = a->hierarchy_bone[h], parent = a->hierarchy_parent[h];
        int relative = -1;
        for (uint32_t i = 0; i < a->relative_count; ++i)
            if (a->relative_bone[i] == bone) relative = (int)i;
        if (relative >= 0) {
            multiply(a->relative[relative], b[parent], b[bone]);
            continue;
        }
        const float *o = a->hierarchy_offset[h], *pm = b[parent];
        for (unsigned c = 0; c < 3; ++c)
            b[bone][12+c] = o[0]*pm[c] + o[1]*pm[4+c] + o[2]*pm[8+c] + pm[12+c];
    }
    for (uint32_t i = 0; i < a->relative_count; ++i) {
        bool placed = false;
        for (uint32_t h = 0; h < a->hierarchy_count; ++h) placed |= a->hierarchy_bone[h] == a->relative_bone[i];
        if (!placed) multiply(a->relative[i], b[a->relative_parent[i]], b[a->relative_bone[i]]);
    }
    static const unsigned links[4][3] = {{26,16,8}, {27,20,14}, {24,5,4}, {25,11,10}};
    for (unsigned i = 0; i < 4; ++i) {
        float s[16];
        swing(b[links[i][1]], b[links[i][2]], s);
        multiply(b[links[i][1]], s, b[links[i][0]]);
        memcpy(b[links[i][0]]+12, b[links[i][2]]+12, 4*sizeof(float));
    }
    blend(b[8], b[26], 0.5f, b[30]);
    blend(b[14], b[27], 0.5f, b[31]);
    blend(b[4], b[24], 0.5f, b[4]);
    blend(b[10], b[25], 0.5f, b[10]);
    memcpy(b[24], b[4], 64); memcpy(b[25], b[10], 64);
    memcpy(b[28], b[24], 64); memcpy(b[29], b[25], 64);
    memcpy(b[28]+12, b[5]+12, 16); memcpy(b[29]+12, b[11]+12, 16);
}

static bool read_all(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static bool load(Asset *a, unsigned slot, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[4];
    uint32_t head[4];
    bool ok = read_all(f, magic, 4) && memcmp(magic, "DXC1", 4) == 0 &&
        read_all(f, head, sizeof head) && head[0] >= 2u && head[0] <= 4u && head[1] <= 16u;
    if (ok) {
        a->relative_count = head[1];
        for (uint32_t i = 0; ok && i < head[1]; ++i)
            ok = read_all(f, &a->relative_bone[i], 4) && read_all(f, &a->relative_parent[i], 4) &&
                read_all(f, a->relative[i], 64) &&
                a->relative_bone[i] < DOAX_BONES && a->relative_parent[i] < DOAX_BONES;
        ok = ok && read_all(f, &a->hierarchy_count, 4) && a->hierarchy_count <= 32u;
        for (uint32_t i = 0; ok && i < a->hierarchy_count; ++i)
            ok = read_all(f, &a->hierarchy_bone[i], 4) && read_all(f, &a->hierarchy_parent[i], 4) &&
                read_all(f, a->hierarchy_offset[i], 12) &&
                a->hierarchy_bone[i] < DOAX_BONES && a->hierarchy_parent[i] < DOAX_BONES;
        a->texture_count = head[2];
        a->textures = calloc(head[2] ? head[2] : 1, sizeof(Texture));
        for (uint32_t i = 0; ok && i < head[2]; ++i) {
            uint32_t t[9];
            Texture *x = &a->textures[i];
            ok = read_all(f, t, sizeof t);
            if (!ok) break;
            x->desc.format_byte = t[0]; x->desc.width = t[1]; x->desc.height = t[2];
            x->desc.mip_levels = t[3]; x->desc.bits_per_pixel = t[4]; x->desc.linear = t[5] != 0;
            x->desc.pitch = t[6];
            /* Outside guest RAM and its aliases, so it cannot collide with a game texture key. */
            x->desc.data = 0xd0000000u + slot * 0x08000000u + i * 0x00100000u;
            x->byte_count = t[7]; x->palette_count = t[8];
            x->bytes = malloc(t[7] ? t[7] : 1); x->palette = t[8] ? malloc(t[8]) : NULL;
            ok = read_all(f, x->bytes, t[7]) && (!t[8] || read_all(f, x->palette, t[8]));
        }
        a->part_count = head[3];
        a->parts = calloc(head[3] ? head[3] : 1, sizeof(Part));
        for (uint32_t i = 0; ok && i < head[3]; ++i) {
            Part *p = &a->parts[i];
            uint32_t h[8];
            ok = read_all(f, h, sizeof h) && h[7] >= 1u && h[7] <= MAX_WORLDS;
            if (!ok) break;
            p->stride = h[0]; p->fvf = h[1]; p->primitive = h[2]; p->vertex_count = h[3];
            p->index_count = h[4]; p->texture = (int32_t)h[5]; p->flags = h[6]; p->world_count = h[7];
            for (uint32_t w = 0; ok && w < p->world_count; ++w)
                ok = read_all(f, &p->bones[w], 4) && read_all(f, p->offsets[w], 64) && p->bones[w] < DOAX_BONES;
            if (ok && head[0] >= 3u) {
                uint32_t present_flag = 0;
                ok = read_all(f, &present_flag, 4) && read_all(f, p->material, sizeof p->material);
                p->has_material = present_flag != 0;
            }
            if (ok && head[0] >= 4u && (p->flags & 4u)) {
                p->soft = true;
                p->mask = malloc((size_t)p->vertex_count * sizeof(float) + 4);
                ok = read_all(f, p->soft_params, sizeof p->soft_params) &&
                    read_all(f, p->mask, (size_t)p->vertex_count * sizeof(float));
            }
            p->vertices = malloc((size_t)p->vertex_count * p->stride + 1);
            p->indices = malloc((size_t)p->index_count * 2 + 2);
            p->skinned = malloc((size_t)p->vertex_count * 32 + 32);
            ok = ok && read_all(f, p->vertices, (size_t)p->vertex_count * p->stride) &&
                read_all(f, p->indices, (size_t)p->index_count * 2) &&
                (p->texture < 0 || (uint32_t)p->texture < a->texture_count);
        }
    }
    fclose(f);
    if (ok) {
        /* Opaque parts first so blended hair and lashes composite over them. */
        for (uint32_t i = 0; i < a->part_count; ++i)
            for (uint32_t j = i + 1; j < a->part_count; ++j)
                if ((a->parts[i].flags & 1u) > (a->parts[j].flags & 1u)) {
                    Part t = a->parts[i]; a->parts[i] = a->parts[j]; a->parts[j] = t;
                }
    }
    return ok;
}

bool recomp_costume_inject_enabled(void)
{
    if (!inject.configured) {
        inject.configured = true;
        const char *both = getenv("RECOMP_COSTUME_INJECT");
        const char *mask = getenv("RECOMP_COSTUME_FIGHTERS");
        inject.fighter_mask = mask ? (unsigned)strtoul(mask, NULL, 0) : 3u;
        const char *names[FIGHTERS] = {"RECOMP_COSTUME_INJECT_P1", "RECOMP_COSTUME_INJECT_P2"};
        for (unsigned f = 0; f < FIGHTERS; ++f) {
            inject.injected_present[f] = UINT32_MAX;
            const char *path = getenv(names[f]);
            if (!path || !*path) path = both;
            if (!path || !*path || !(inject.fighter_mask & (1u << f))) continue;
            /* Both fighters share one load when they name the same file. */
            if (f == 1 && inject.fighter_asset[0] && both && path == both) {
                inject.fighter_asset[1] = inject.fighter_asset[0];
                continue;
            }
            Asset *a = &inject.assets[f];
            bool ok = load(a, f, path);
            fprintf(stderr, "recomp costume inject: fighter %u %s %s (%u parts, %u textures)\n",
                f + 1, ok ? "loaded" : "failed to load", path, a->part_count, a->texture_count);
            if (ok) { inject.fighter_asset[f] = a; inject.enabled = true; }
        }
    }
    return inject.enabled;
}

static void skin(Part *p, float bank[DOAX_BONES][16], const float *soft_offset)
{
    RecompD3dVertexLayout layout;
    if (!recomp_d3d_fvf_layout(p->fvf, &layout)) { p->vertex_count = 0; return; }
    float m[MAX_WORLDS][16];
    for (uint32_t w = 0; w < p->world_count; ++w) multiply(p->offsets[w], bank[p->bones[w]], m[w]);
    const uint32_t weights = layout.blend_weight_count < p->world_count ? layout.blend_weight_count : p->world_count - 1;
    for (uint32_t v = 0; v < p->vertex_count; ++v) {
        const uint8_t *src = p->vertices + (size_t)v * p->stride;
        float pos[3], nrm[3] = {0, 0, 0}, uv[2] = {0, 0}, wt[MAX_WORLDS] = {1, 0, 0, 0};
        memcpy(pos, src, 12);
        if (layout.normal_offset != RECOMP_D3D_FVF_ABSENT) memcpy(nrm, src + layout.normal_offset, 12);
        if (layout.texcoord_offset != RECOMP_D3D_FVF_ABSENT) memcpy(uv, src + layout.texcoord_offset, 8);
        if (weights) {
            float rest = 1;
            memcpy(wt, src + 12, weights * 4);
            for (uint32_t k = 0; k < weights; ++k) rest -= wt[k];
            wt[weights] = rest;
        }
        float *out = p->skinned + (size_t)v * 8;
        memset(out, 0, 6 * sizeof(float));
        for (uint32_t k = 0; k <= weights; ++k) {
            const float *a = m[k];
            for (unsigned c = 0; c < 3; ++c) {
                out[c] += wt[k] * (pos[0]*a[c] + pos[1]*a[4+c] + pos[2]*a[8+c] + a[12+c]);
                out[3+c] += wt[k] * (nrm[0]*a[c] + nrm[1]*a[4+c] + nrm[2]*a[8+c]);
            }
        }
        if (soft_offset && p->mask[v] > 0.0f) {
            const float *r = bank[p->bones[0]];
            for (unsigned c = 0; c < 3; ++c)
                out[c] += p->mask[v] * (soft_offset[0]*r[c] + soft_offset[1]*r[4+c] + soft_offset[2]*r[8+c]);
        }
        float l = sqrtf(out[3]*out[3] + out[4]*out[4] + out[5]*out[5]);
        if (l > 0) for (unsigned c = 3; c < 6; ++c) out[c] /= l;
        out[6] = uv[0]; out[7] = uv[1];
    }
}

/* The template's lit material is emissive + ambient * scene ambient (plus
   per-light terms). Recover the scene terms through the template's own raw
   material and apply them to this part's source material. */
static float ratio(float value, float template_term, float part_term)
{
    return fabsf(template_term) > 1e-4f ? value / template_term * part_term : part_term * 0.0f;
}

static void relight(RecompD3dDirectionalLighting *l, const float *t, const Part *p)
{
    if (!l->enabled || !t || !p->has_material) return;
    for (unsigned c = 0; c < 3; ++c) {
        float scene = fabsf(t[4 + c]) > 1e-4f ? (l->ambient_emissive[c] - t[12 + c]) / t[4 + c] : 0.0f;
        l->ambient_emissive[c] = p->material[12 + c] + p->material[4 + c] * scene;
        for (unsigned i = 0; i < l->count; ++i)
            l->ambient[i][c] = ratio(l->ambient[i][c], t[4 + c], p->material[4 + c]);
        for (unsigned i = 0; i < l->count; ++i)
            l->specular[i][c] = ratio(l->specular[i][c], t[8 + c], p->material[8 + c]);
    }
    memcpy(l->material_diffuse, p->material, 4 * sizeof(float));
}

/* One damped spring per soft part, in its bone's frame, pushed opposite to
   the bone's acceleration (inertia). Two substeps per present at 60 Hz. */
static void step_soft(unsigned f, Asset *a, float bank[DOAX_BONES][16], uint32_t present)
{
    if (!inject.soft_state[f]) inject.soft_state[f] = calloc((size_t)a->part_count * 6u + 6u, sizeof(float));
    if (inject.history_present[f] + 1u != present) inject.history_count[f] = 0;
    float (*h)[DOAX_BONES][3] = inject.history[f];
    const float dt = 1.0f / 60.0f;
    static float scale = -1.0f;
    if (scale < 0.0f) {
        const char *text = getenv("RECOMP_COSTUME_JIGGLE");
        scale = text ? (float)atof(text) : 0.7f;
        if (!(scale >= 0.0f)) scale = 1.0f;
    }
    for (uint32_t i = 0; i < a->part_count; ++i) {
        Part *p = &a->parts[i];
        if (!p->soft) continue;
        float *s = inject.soft_state[f] + i * 6u, *v = s + 3;
        const uint32_t b = p->bones[0];
        float accel[3] = {0, 0, 0}, local[3];
        if (inject.history_count[f] >= 2u)
            for (unsigned c = 0; c < 3; ++c)
                accel[c] = (bank[b][12+c] - 2.0f*h[1][b][c] + h[0][b][c]) / (dt*dt);
        for (unsigned c = 0; c < 3; ++c)
            local[c] = accel[0]*bank[b][c*4] + accel[1]*bank[b][c*4+1] + accel[2]*bank[b][c*4+2];
        const float k = p->soft_params[0], damping = p->soft_params[1], gain = p->soft_params[2] * scale, limit = p->soft_params[3];
        for (unsigned step = 0; step < 2; ++step) {
            const float t = dt * 0.5f;
            for (unsigned c = 0; c < 3; ++c) {
                v[c] += (-k*s[c] - damping*v[c] - gain*local[c]) * t;
                s[c] += v[c] * t;
            }
        }
        float len = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
        if (len > limit) for (unsigned c = 0; c < 3; ++c) { s[c] *= limit / len; v[c] *= 0.5f; }
    }
    if (getenv("RECOMP_COSTUME_DEBUG") && present % 20u == 0u) {
        float hair = 0, chest = 0;
        for (uint32_t i = 0; i < a->part_count; ++i) {
            if (!a->parts[i].soft) continue;
            const float *s = inject.soft_state[f] + i * 6u;
            float len = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
            if (a->parts[i].bones[0] == 1u) hair = fmaxf(hair, len); else chest = fmaxf(chest, len);
        }
        fprintf(stderr, "recomp costume physics: present=%u fighter=%u hair=%.4f chest=%.4f\n", present, f, hair, chest);
    }
    memcpy(h[0], h[1], sizeof h[0]);
    for (unsigned b = 0; b < DOAX_BONES; ++b) memcpy(h[1][b], bank[b] + 12, 3 * sizeof(float));
    if (inject.history_count[f] < 2u) ++inject.history_count[f];
    inject.history_present[f] = present;
}

static void submit_fighter(unsigned fighter, Asset *a, const float *template_material, float bank[DOAX_BONES][16], const float view[16], const float projection[16],
    const RecompD3dPresenterDrawCommand *template_draw)
{
    float view_projection[16];
    multiply(view, projection, view_projection);
    static const float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    for (uint32_t i = 0; i < a->part_count; ++i) {
        Part *p = &a->parts[i];
        if (!p->vertex_count || !p->index_count) continue;
        skin(p, bank, p->soft && inject.soft_state[fighter] ? inject.soft_state[fighter] + i * 6u : NULL);
        RecompD3dPresenterCommand command;
        memset(&command, 0, sizeof command);
        command.type = RECOMP_D3D_PRESENTER_COMMAND_DRAW;
        RecompD3dPresenterDrawCommand *d = &command.data.draw;
        *d = *template_draw;
        d->primitive_type = p->primitive;
        d->index_count = p->index_count;
        d->triangle_count = recomp_d3d_draw_triangle_count(p->primitive, p->index_count);
        d->vertex_count = p->vertex_count;
        d->vertex_stride = 32;
        d->fvf = 0x112u;
        d->program_count = 0;
        d->vertex_bytes = p->skinned;
        d->index_bytes = p->indices;
        memcpy(d->transform, view_projection, sizeof view_projection);
        d->has_transform = true;
        d->blend_weight_count = 0;
        d->cull_mode = RECOMP_D3D_CULL_NONE;
        if (d->directional.enabled) {
            memcpy(d->directional.world_transforms[0], identity, sizeof identity);
            memcpy(d->directional.normal_transforms[0], identity, sizeof identity);
            relight(&d->directional, template_material, p);
        }
        d->has_alpha_mask = false;
        d->has_reflection = false;
        d->use_texture_factor = false;
        d->modulate_texture_factor = false;
        d->texture_is_backbuffer = false;
        d->texture_is_front_buffer = false;
        d->has_texture = p->texture >= 0;
        if (d->has_texture) {
            const Texture *t = &a->textures[p->texture];
            d->texture = t->desc;
            d->texture_bytes = t->bytes;
            d->texture_byte_count = t->byte_count;
            d->palette_bytes = t->palette;
            d->palette_byte_count = t->palette_count;
        }
        if (p->flags & 1u) {
            d->blend.blend_enable = true;
            d->blend.src_factor = RECOMP_D3D_BLEND_SRC_ALPHA;
            d->blend.dst_factor = RECOMP_D3D_BLEND_INV_SRC_ALPHA;
            d->blend.op = RECOMP_D3D_BLEND_OP_ADD;
        } else {
            d->blend.blend_enable = false;
        }
        d->blend.color_write_mask = 15;
        d->depth.depth_write_enable = (p->flags & 2u) != 0;
        recomp_d3d_presenter_submit(recomp_d3d_frame_adapter_presenter(), &command);
    }
}

static void note_span(unsigned f, uint32_t present, uint32_t vertex_data)
{
    if (!vertex_data) return;
    if (inject.span_present[f] != present) {
        if (inject.span_present[f] != 0u) {
            inject.last_low[f] = inject.span_low[f]; inject.last_high[f] = inject.span_high[f];
        }
        inject.span_present[f] = present;
        inject.span_low[f] = vertex_data; inject.span_high[f] = vertex_data;
    }
    if (vertex_data < inject.span_low[f]) inject.span_low[f] = vertex_data;
    if (vertex_data > inject.span_high[f]) inject.span_high[f] = vertex_data;
}

bool recomp_costume_inject_draw(uint32_t present, uint32_t vertex_data, const float world[16],
    const float view[16], const float projection[16],
    const RecompD3dPresenterDrawCommand *template_draw,
    const float *template_material)
{
    if (!recomp_costume_inject_enabled()) return false;
    for (unsigned f = 0; f < FIGHTERS; ++f) {
        if (!inject.fighter_asset[f]) continue;
        float bones[DOA3_BONES][16];
        recomp_guest_load(bones, bank_base[f], sizeof bones);
        bool match = false, attached = false;
        for (unsigned i = 0; i < DOA3_BONES && !match; ++i) match = memcmp(bones[i], world, 64) == 0;
        /* Props hung on a bone (ribbons, shoe straps) share its rotation at an offset. */
        for (unsigned i = 0; i < DOA3_BONES && !match && !attached; ++i) {
            float err = 0, dist = 0;
            for (unsigned k = 0; k < 12; ++k) err = fmaxf(err, fabsf(bones[i][k] - world[k]));
            for (unsigned k = 12; k < 15; ++k) dist += (bones[i][k] - world[k]) * (bones[i][k] - world[k]);
            attached = err < 1e-4f && dist < 0.25f && fabsf(bones[i][0]) + fabsf(bones[i][5]) + fabsf(bones[i][10]) < 2.999f;
        }
        if (!match) {
            if (attached) return true;
            /* Same model allocation as last present's matched draws, and near the fighter. */
            if (vertex_data && inject.last_high[f] > inject.last_low[f] &&
                vertex_data >= inject.last_low[f] && vertex_data <= inject.last_high[f]) {
                for (unsigned i = 0; i < DOA3_BONES; ++i) {
                    float d = 0;
                    for (unsigned k = 12; k < 15; ++k) d += (bones[i][k]-world[k])*(bones[i][k]-world[k]);
                    if (d < 0.25f) return true;
                }
            }
            continue;
        }
        note_span(f, present, vertex_data);
        if (template_draw->directional.enabled && template_material) {
            inject.lit[f] = template_draw->directional;
            memcpy(inject.lit_material[f], template_material, sizeof inject.lit_material[f]);
            inject.have_lit[f] = true;
        }
        if (inject.injected_present[f] != present) {
            inject.injected_present[f] = present;
            RecompD3dPresenterDrawCommand lit_template;
            if (!template_draw->directional.enabled && inject.have_lit[f]) {
                lit_template = *template_draw;
                lit_template.directional = inject.lit[f];
                template_draw = &lit_template;
                template_material = inject.lit_material[f];
            }
            float bank[DOAX_BONES][16];
            doax_bank(inject.fighter_asset[f], bones, bank);
            step_soft(f, inject.fighter_asset[f], bank, present);
            submit_fighter(f, inject.fighter_asset[f], template_material, bank, view, projection, template_draw);
        }
        return true;
    }
    return false;
}
