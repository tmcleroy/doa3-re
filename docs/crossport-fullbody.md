# DOAXBV girls in DOA3: whole-character port (paused)

This branch holds the DOA3 side of the paused whole-character port: the mesh
dump (`recomp-runtime/mesh_dump.{c,h}`), the DOAXBV-girl injector
(`recomp-runtime/costume_inject.{c,h}`, F5/F6 cycling, `RECOMP_COSTUME_*`
switches) and the launcher's Player 1/Player 2 costume pickers.

The methodology, converters and checking tools are on the doaxvb-re branch of
the same name: `docs/crossport-fullbody.md` and `tools/crossport/`. Assets
(`private/doax-*.dxc`) are built locally from a user-owned disc and never
committed.
