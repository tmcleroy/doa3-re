#include "mesh_dump.h"
#include "runtime.h"

#include <SDL3/SDL.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define mesh_dump_mkdir(path) _mkdir(path)
#else
#define mesh_dump_mkdir(path) mkdir(path, 0755)
#endif

enum { MESH_DUMP_PRESENTS = 16, MESH_DUMP_TEXTURES = 4096 };

static struct {
    bool configured, enabled;
    const char *root, *trigger;
    uint32_t at[MESH_DUMP_PRESENTS], at_count;
    uint32_t checked_present, triggered_present;
    uint32_t open_present;
    FILE *index;
    char folder[1024];
    uint32_t ordinal;
    uint32_t textures[MESH_DUMP_TEXTURES], texture_count;
    /* F9 captures a burst of presents (with RAM) into capture-NN/. */
    uint32_t hotkey_present, burst[MESH_DUMP_PRESENTS], burst_count, capture;
    bool hotkey_down;
} dump;

enum { MESH_DUMP_BURST = 8, MESH_DUMP_BURST_SPACING = 20 };

static bool in_burst(uint32_t present)
{
    for (uint32_t i = 0u; i < dump.burst_count; ++i)
        if (dump.burst[i] == present) return true;
    return false;
}

static void configure(void)
{
    dump.configured = true;
    dump.root = getenv("RECOMP_MESH_DUMP");
    if (dump.root == NULL || *dump.root == '\0') return;
    dump.trigger = getenv("RECOMP_MESH_DUMP_TRIGGER");
    const char *at = getenv("RECOMP_MESH_DUMP_AT");
    while (at != NULL && *at != '\0' && dump.at_count < MESH_DUMP_PRESENTS) {
        char *end;
        errno = 0;
        unsigned long value = strtoul(at, &end, 10);
        if (errno != 0 || end == at || value == 0ul || value > UINT32_MAX) break;
        dump.at[dump.at_count++] = (uint32_t)value;
        at = *end == ',' ? end + 1 : end;
    }
    dump.enabled = true;
    dump.checked_present = dump.triggered_present = dump.open_present = UINT32_MAX;
    dump.hotkey_present = UINT32_MAX;
    fprintf(stderr, "recomp mesh dump: press F9 to capture %u frames into %s/capture-NN\n",
        (unsigned)MESH_DUMP_BURST, dump.root);
    if (dump.enabled) mesh_dump_mkdir(dump.root);
}

bool recomp_mesh_dump_active(uint32_t present)
{
    if (!dump.configured) configure();
    if (!dump.enabled) return false;
    if (dump.hotkey_present != present) {
        dump.hotkey_present = present;
        const bool *keys = SDL_GetKeyboardState(NULL);
        bool down = keys != NULL && keys[SDL_SCANCODE_F9];
        if (down && !dump.hotkey_down) {
            ++dump.capture;
            dump.burst_count = MESH_DUMP_BURST;
            for (uint32_t i = 0u; i < MESH_DUMP_BURST; ++i)
                dump.burst[i] = present + 1u + i * MESH_DUMP_BURST_SPACING;
            fprintf(stderr, "recomp mesh dump: capture %u started\n", dump.capture);
        }
        dump.hotkey_down = down;
    }
    if (in_burst(present)) return true;
    for (uint32_t i = 0u; i < dump.at_count; ++i)
        if (dump.at[i] == present) return true;
    if (dump.trigger != NULL && dump.checked_present != present) {
        struct stat info;
        dump.checked_present = present;
        if (stat(dump.trigger, &info) == 0 && remove(dump.trigger) == 0)
            dump.triggered_present = present;
    }
    return dump.triggered_present == present;
}

static void write_file(const char *name, const void *bytes, size_t size);

static bool open_present(uint32_t present)
{
    if (dump.open_present == present) return dump.index != NULL;
    if (dump.index != NULL) fclose(dump.index);
    dump.index = NULL;
    dump.open_present = present;
    dump.ordinal = 0u;
    dump.texture_count = 0u;
    if (in_burst(present)) {
        char capture[900];
        snprintf(capture, sizeof capture, "%s/capture-%02u", dump.root, dump.capture);
        mesh_dump_mkdir(capture);
        snprintf(dump.folder, sizeof dump.folder, "%s/present-%u", capture, present);
    } else {
        snprintf(dump.folder, sizeof dump.folder, "%s/present-%u", dump.root, present);
    }
    mesh_dump_mkdir(dump.folder);
    char path[1100];
    snprintf(path, sizeof path, "%s/draws.jsonl", dump.folder);
    dump.index = fopen(path, "wb");
    if (dump.index != NULL)
        fprintf(stderr, "recomp mesh dump: writing %s\n", dump.folder);
    /* Optional guest RAM snapshot, taken before this present's first draw, so
       bone matrices can be located by value offline. Private data. */
    const char *ram = getenv("RECOMP_MESH_DUMP_RAM");
    if (in_burst(present) || (ram != NULL && *ram != '\0' && *ram != '0')) {
        const uint8_t *bytes = recomp_memory(0u, RECOMP_XBOX_RAM_SIZE);
        if (bytes != NULL) write_file("ram.bin", bytes, RECOMP_XBOX_RAM_SIZE);
    }
    return dump.index != NULL;
}

static void write_file(const char *name, const void *bytes, size_t size)
{
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", dump.folder, name);
    FILE *file = fopen(path, "wb");
    if (file == NULL) return;
    fwrite(bytes, 1, size, file);
    fclose(file);
}

static void write_matrix(const char *name, const float m[16])
{
    fprintf(dump.index, ",\"%s\":[", name);
    for (unsigned i = 0u; i < 16u; ++i)
        fprintf(dump.index, "%s%.9g", i ? "," : "", m[i]);
    fputc(']', dump.index);
}

/* Each distinct texture is written once per present, named by guest address. */
static void write_texture(const RecompD3dPresenterDrawCommand *draw)
{
    const RecompD3dTextureDesc *t = &draw->texture;
    fprintf(dump.index,
        ",\"texture\":{\"data\":%u,\"format\":%u,\"bpp\":%u,\"linear\":%s,"
        "\"width\":%u,\"height\":%u,\"pitch\":%u,\"mips\":%u,\"bytes\":%u,"
        "\"palette_bytes\":%u}",
        t->data, t->format_byte, t->bits_per_pixel, t->linear ? "true" : "false",
        t->width, t->height, t->pitch, t->mip_levels, draw->texture_byte_count,
        draw->palette_bytes != NULL ? draw->palette_byte_count : 0u);
    if (draw->texture_bytes == NULL || draw->texture_byte_count == 0u ||
        draw->texture_is_backbuffer || draw->texture_is_front_buffer) return;
    for (uint32_t i = 0u; i < dump.texture_count; ++i)
        if (dump.textures[i] == t->data) return;
    if (dump.texture_count == MESH_DUMP_TEXTURES) return;
    dump.textures[dump.texture_count++] = t->data;
    char name[64];
    snprintf(name, sizeof name, "tex-%08x.bin", t->data);
    write_file(name, draw->texture_bytes, draw->texture_byte_count);
    if (draw->palette_bytes != NULL && draw->palette_byte_count != 0u) {
        snprintf(name, sizeof name, "pal-%08x.bin", t->data);
        write_file(name, draw->palette_bytes, draw->palette_byte_count);
    }
}

void recomp_mesh_dump_draw(uint32_t present,
    const RecompD3dPresenterDrawCommand *draw,
    uint32_t vertex_data, uint32_t index_data,
    const float worlds[4][16], uint32_t world_count,
    const float view[16], const float projection[16],
    const float *material)
{
    if (draw == NULL || !recomp_mesh_dump_active(present) || !open_present(present)) return;
    const uint32_t ordinal = dump.ordinal++;
    const size_t vertex_size = (size_t)draw->vertex_count * draw->vertex_stride;
    const size_t index_size = (size_t)draw->index_count * 2u;
    char name[64];

    if (draw->vertex_bytes != NULL && vertex_size != 0u) {
        snprintf(name, sizeof name, "d%05u.vtx", ordinal);
        write_file(name, draw->vertex_bytes, vertex_size);
    }
    if (draw->index_bytes != NULL && index_size != 0u) {
        snprintf(name, sizeof name, "d%05u.idx", ordinal);
        write_file(name, draw->index_bytes, index_size);
    }
    fprintf(dump.index,
        "{\"ordinal\":%u,\"primitive\":%u,\"index_count\":%u,\"triangles\":%u,"
        "\"vertex_count\":%u,\"stride\":%u,\"fvf\":%u,\"vertex_data\":%u,"
        "\"index_data\":%u,\"blend_weights\":%u,\"program\":%u,\"offscreen\":%s,"
        "\"cull\":%u,\"alpha_blend\":%s,\"color_mask\":%u,\"stencil\":%s,"
        "\"depth_write\":%s",
        ordinal, draw->primitive_type, draw->index_count, draw->triangle_count,
        draw->vertex_count, draw->vertex_stride, draw->fvf, vertex_data, index_data,
        draw->blend_weight_count, draw->program_count,
        draw->target.offscreen ? "true" : "false", (unsigned)draw->cull_mode,
        draw->blend.blend_enable ? "true" : "false",
        (unsigned)draw->blend.color_write_mask,
        draw->depth.stencil_enable ? "true" : "false",
        draw->depth.depth_write_enable ? "true" : "false");
    fputs(",\"worlds\":[", dump.index);
    for (uint32_t i = 0u; i < world_count; ++i) {
        fputs(i ? ",[" : "[", dump.index);
        for (unsigned k = 0u; k < 16u; ++k)
            fprintf(dump.index, "%s%.9g", k ? "," : "", worlds[i][k]);
        fputc(']', dump.index);
    }
    fputc(']', dump.index);
    if (view != NULL) write_matrix("view", view);
    if (projection != NULL) write_matrix("projection", projection);
    write_matrix("transform", draw->transform);
    if (draw->has_texture) write_texture(draw);
    if (material != NULL) {
        fputs(",\"raw_material\":[", dump.index);
        for (unsigned i = 0u; i < 17u; ++i) fprintf(dump.index, "%s%.9g", i ? "," : "", material[i]);
        fputc(']', dump.index);
    }
    fprintf(dump.index,
        ",\"material\":{\"lit\":%s,\"diffuse\":[%.6g,%.6g,%.6g,%.6g],"
        "\"ambient_emissive\":[%.6g,%.6g,%.6g,%.6g],\"texture_factor\":%u,"
        "\"use_texture_factor\":%s,\"modulate_texture_factor\":%s,"
        "\"material_alpha_mode\":%u,\"material_alpha\":%.6g,"
        "\"blend_src\":%u,\"blend_dst\":%u,\"blend_op\":%u,\"alpha_test\":%s,"
        "\"alpha_func\":%u,\"alpha_ref\":%u,\"z_func\":%u}",
        draw->directional.enabled ? "true" : "false",
        draw->directional.material_diffuse[0], draw->directional.material_diffuse[1],
        draw->directional.material_diffuse[2], draw->directional.material_diffuse[3],
        draw->directional.ambient_emissive[0], draw->directional.ambient_emissive[1],
        draw->directional.ambient_emissive[2], draw->directional.ambient_emissive[3],
        draw->texture_factor, draw->use_texture_factor ? "true" : "false",
        draw->modulate_texture_factor ? "true" : "false",
        (unsigned)draw->material_alpha_mode, draw->material_alpha,
        (unsigned)draw->blend.src_factor, (unsigned)draw->blend.dst_factor,
        (unsigned)draw->blend.op, draw->depth.alpha_test_enable ? "true" : "false",
        (unsigned)draw->depth.alpha_func, draw->depth.alpha_ref, (unsigned)draw->depth.depth_func);
    fputs("}\n", dump.index);
    fflush(dump.index);
}
