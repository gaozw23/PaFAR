from pafar_sim.cli import main
raise SystemExit(main(["aggregate", *__import__("sys").argv[1:]]))

