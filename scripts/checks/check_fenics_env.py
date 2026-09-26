try:
    import dolfin
    print("dolfin", dolfin.__version__)
except ImportError as e:
    print("NO dolfin:", e)
try:
    import mshr
    print("mshr ok")
except ImportError as e:
    print("NO mshr:", e)