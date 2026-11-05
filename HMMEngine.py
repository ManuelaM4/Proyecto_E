"""
HMMEngine.py
Capa matemática (núcleo). Implementación nativa de HMM con solo NumPy.
No conoce trazos, letras, archivos ni interfaz: solo matrices y símbolos.
"""
import numpy as np

EPS = 1e-12

class HMM:
    """HMM discreto lambda = (A, B, pi) con topología left-right."""

    def __init__(self, n_states, n_symbols, seed=0):
        self.N = n_states
        self.K = n_symbols
        self.mean_len = 0
        
        # Matriz de Transición A (Topología Left-Right)
        A = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            A[i, i] = 0.6
            A[i, i + 1] = 0.4
        A[self.N - 1, self.N - 1] = 1.0
        self.A = A

        # Matriz de Emisión B (Inicialización Uniforme para estabilidad)
        self.B = np.ones((self.N, self.K)) / self.K

        # Vector Inicial pi (Siempre empieza en el estado 0)
        self.pi = np.zeros(self.N)
        self.pi[0] = 1.0

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)
        alpha[0] = self.pi * self.B[:, O[0]]
        c[0] = alpha[0].sum() + EPS
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, O[t]]
            c[t] = alpha[t].sum() + EPS
            alpha[t] /= c[t]
        return alpha, c

    def backward(self, O, c):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, O[t + 1]] * beta[t + 1]) / c[t + 1]
        return beta

    def log_likelihood(self, O):
        if len(O) == 0:
            return 0.0
        _, c = self.forward(O)
        return float(np.sum(np.log(c)))

    def viterbi(self, O):
        T = len(O)
        with np.errstate(divide="ignore"):
            logA = np.log(self.A)
            logB = np.log(self.B)
            logpi = np.log(self.pi)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, O[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.N)] + logB[:, O[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(np.max(delta[-1]))

    def baum_welch(self, sequences, n_iter=50, tol=1e-4, verbose=False):
        sequences = [np.asarray(s, dtype=int) for s in sequences if len(s) > 1]
        if not sequences: return 0
        self.mean_len = int(round(np.mean([len(s) for s in sequences])))
        prev_ll = -np.inf
        total_ll = prev_ll

        for it in range(n_iter):
            num_A = np.zeros((self.N, self.N))
            den_A = np.zeros(self.N)
            num_B = np.zeros((self.N, self.K))
            den_B = np.zeros(self.N)
            pi_acc = np.zeros(self.N)
            total_ll = 0.0

            for O in sequences:
                alpha, c = self.forward(O)
                beta = self.backward(O, c)
                total_ll += np.sum(np.log(c))

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + EPS

                b_next = self.B[:, O[1:]].T * beta[1:]
                xi = alpha[:-1, :, None] * self.A[None, :, :] * b_next[:, None, :]
                xi /= c[1:, None, None]
                num_A += xi.sum(axis=0)
                den_A += gamma[:-1].sum(axis=0)

                for k in range(self.K):
                    num_B[:, k] += gamma[O == k].sum(axis=0)
                den_B += gamma.sum(axis=0)
                pi_acc += gamma[0]

            A_new = num_A / (den_A[:, None] + EPS)
            A_new = np.where(self.A > 0, A_new + EPS, 0.0)
            self.A = A_new / A_new.sum(axis=1, keepdims=True)

            B_new = num_B / (den_B[:, None] + EPS) + 1e-6
            self.B = B_new / B_new.sum(axis=1, keepdims=True)

            self.pi = pi_acc / (pi_acc.sum() + EPS)

            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll
        return total_ll

    def predict_future(self, O, total_len=None):
        T_total = total_len or self.mean_len
        t = len(O)
        if t == 0:
            p = self.pi.copy()
        else:
            alpha, _ = self.forward(O)
            p = alpha[-1].copy()
        future = []
        for _ in range(t, T_total):
            p = p @ self.A
            p /= p.sum() + EPS
            future.append(int(np.argmax(p @ self.B)))
        return future

    def to_dict(self):
        return {"N": self.N, "K": self.K, "mean_len": self.mean_len,
                "A": self.A.tolist(), "B": self.B.tolist(), "pi": self.pi.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["N"], d["K"])
        m.A = np.array(d["A"])
        m.B = np.array(d["B"])
        m.pi = np.array(d["pi"])
        m.mean_len = d["mean_len"]
        return m


class HMMClassifier:
    def __init__(self, models=None):
        self.models = models or {}

    def posteriors(self, O):
        chars = list(self.models.keys())
        logl = np.array([self.models[c].log_likelihood(O) for c in chars])
        logl -= logl.max()
        p = np.exp(logl)
        p /= p.sum()
        order = np.argsort(-p)
        return [(chars[i], float(p[i])) for i in order]

    def best_char(self, O):
        return self.posteriors(O)[0][0]

    def project(self, O):
        c = self.best_char(O)
        return c, self.models[c].predict_future(O)