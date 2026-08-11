from pafar_sim.cli import main
raise SystemExit(main(["build-oracle", *__import__("sys").argv[1:]]))

