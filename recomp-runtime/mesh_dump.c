#include "mesh_dump.h"

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
} dump;

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
    dump.enabled = dump.at_count != 0u || dump.trigger != NULL;
    dump.checked_present = dump.triggered_present = dump.open_present = UINT32_MAX;
    if (dump.enabled) mesh_dump_mkdir(dump.root);
}

bool recomp_mesh_dump_active(uint32_t present)
{
    if (!dump.configured) configure();
    if (!dump.enabled) return false;
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

static bool open_present(uint32_t present)
{
    if (dump.open_present == present) return dump.index != NULL;
    if (dump.index != NULL) fclose(dump.index);
    dump.index = NULL;
    dump.open_present = present;
    dump.ordinal = 0u;
    dump.texture_count = 0u;
    snprintf(dump.folder, sizeof dump.folder, "%s/present-%u", dump.root, present);
    mesh_dump_mkdir(dump.folder);
    char path[1100];
    snprintf(path, sizeof path, "%s/draws.jsonl", dump.folder);
    dump.index = fopen(path, "wb");
    if (dump.index != NULL)
        fprintf(stderr, "recomp mesh dump: writing %s\n", dump.folder);
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
    const float view[16], const float projection[16])
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
    fputs("}\n", dump.index);
    fflush(dump.index);
}
