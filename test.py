import ray
import numpy as np
import atomicarray

norm = np.linalg.norm

@ray.remote 
class Actor():
    def __init__(self, i ):
        self.i = i


    def apply_delta(self, delta):
        for k in self.weights:
            atomicarray.increment(self.weights[k], delta[k])

    def inc(self):
        delta = {k: 0.1 * np.ones_like(self.weights[k]) for k in self.weights}
        self.apply_delta(delta)

    def loop_inc(self, shared, n=00):
        self.weights = shared
        print([(k, norm(w)) for k, w in self.weights.items()])

        for i in range(n):
            self.inc()

        print([(k, norm(w)) for k, w in self.weights.items()])

if __name__ == '__main__':
    ray.init()
    shared_dict = {k: np.random.random_sample((2,2)) for k in range(5)}
    flattened = {k:v.flatten() for k, v in shared_dict.items()}
    sid = ray.put(flattened)
    actors = [Actor.remote(i) for i in range(3)]
    ray.get([a.loop_inc.remote(sid) for a in actors])
    print("Result:", ray.get(sid))
