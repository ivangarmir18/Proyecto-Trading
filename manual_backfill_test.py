from importlib import import_module
om = import_module("core.orchestrator")
# si exporta run_full_backfill_safe
for name in ("run_full_backfill_safe","run_full_backfill"):
    if hasattr(om, name):
        print("Calling", name)
        try:
            r = getattr(om, name)()
            print("Result:", r)
        except Exception as e:
            print("Error calling", name, e)
# Try class-based
if hasattr(om, "Orchestrator"):
    try:
        Or = getattr(om, "Orchestrator")
        inst = Or()
        if hasattr(inst, "run_full_backfill_safe"):
            print("Calling Orchestrator().run_full_backfill_safe()")
            print(inst.run_full_backfill_safe())
    except Exception as e:
        print("Orchestrator class call error", e)
