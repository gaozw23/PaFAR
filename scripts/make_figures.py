from pafar_sim.cli import main
raise SystemExit(main(["figures", *__import__("sys").argv[1:]]))

