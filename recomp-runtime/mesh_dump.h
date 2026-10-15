#ifndef DOA3_RECOMP_MESH_DUMP_H
#define DOA3_RECOMP_MESH_DUMP_H

#include "d3d_presenter.h"

#include <stdbool.h>
#include <stdint.h>

/* Diagnostic mesh dump: writes every indexed draw of one present, with its
   raw vertex and index bytes, world/view/projection and blend worlds, and
   the stage-0 texture, so geometry can be rebuilt offline in world space.

   RECOMP_MESH_DUMP=<dir> enables it. RECOMP_MESH_DUMP_AT=<present>[,...]
   picks presents; RECOMP_MESH_DUMP_TRIGGER=<file> dumps the next present
   after the file appears and deletes it, so a running game can be dumped
   repeatedly. Output goes to <dir>/present-<n>/ and is private data. */

#ifdef __cplusplus
extern "C" {
#endif

/* True when the present about to be drawn is being dumped. Cheap after the
   first call; callers use it to skip gathering matrices on normal frames. */
bool recomp_mesh_dump_active(uint32_t present);

/* worlds holds world_count matrices (WORLD0..WORLD3, row-vector). */
void recomp_mesh_dump_draw(uint32_t present,
    const RecompD3dPresenterDrawCommand *draw,
    uint32_t vertex_data, uint32_t index_data,
    const float worlds[4][16], uint32_t world_count,
    const float view[16], const float projection[16]);

#ifdef __cplusplus
}
#endif

#endif
