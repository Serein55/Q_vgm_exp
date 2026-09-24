"""Use the same evaluation protocol with an offline-trained action expert."""

import sys

from eval_sft import main

if __name__ == "__main__":
    if "--actor-checkpoint" not in sys.argv:
        raise SystemExit("Required: --actor-checkpoint artifacts/<run>/actor_<tag>.pt")
    if "--name" not in sys.argv:
        sys.argv.extend(["--name", "qvgm"])
    main()
