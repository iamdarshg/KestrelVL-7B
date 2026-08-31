# Cost and budget policy

The implementation defaults to a prototype cap of **$90**, below the requested
$100 GCP credit ceiling. A launcher estimates cost before creating a VM and
refuses a cap of $100 or greater. Actual billing is recorded from user-supplied
pricing and wall time in `reports/runtime/cost.jsonl`; pricing assumptions are
not evidence of an invoice.

The first cloud gate is intentionally small: architecture/recovery smoke,
vision projector smoke, and a short throughput comparison. It must establish
whether the transplant works before selective CPT/SFT/RL spend. Full frontier
benchmark attainment cannot be promised within $100; the code keeps the
requested research order and exposes that limitation rather than silently
substituting a toy result.

