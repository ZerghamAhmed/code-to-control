"""Support code for the Code to Control runner (scripts/run_controller_flappy.py).

    abstractions.objapi      the object API that synthesized controllers call
    abstractions.trajectory  formats a played episode for the language model
    llm_client               sends prompts to the language model
    missions                 the generic mission statement given in every game
    runmeta                  run provenance and per-episode instrumentation

The package name comes from TheoryCoder, the codebase this work builds on. Only the modules
the runner imports are included.
"""
