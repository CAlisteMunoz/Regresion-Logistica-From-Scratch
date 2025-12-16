import numpy as np

class reg_log:
    def __init__(self, epsilon=1e-4, max_iter=100_000, eta=0.01, m=10, verbose=True):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.eta = eta
        self.m = m
        self.verbose = verbose

    def sigmoide(self, X: np.array, w: np.array):
        z = X @ w
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def f_costo(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        y_pred = self.sigmoide(X, w)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def f_gradiente(self, X: np.array, w: np.array, y: np.array):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        Sig_f = self.sigmoide(X, w)
        n = X.shape[0]
        gradiente = (X.T @ (Sig_f - y)) / n
        return gradiente

    def _two_loop_recursion(self, grad, s_list, y_list, rho_list):
        # Two-loop recursion: devuelve p = -H_k * grad
        q = grad.copy()
        alpha = []
        for i in range(len(s_list) - 1, -1, -1):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            a = float(rho * (s.T @ q))
            alpha.append(a)
            q = q - a * y
        if len(s_list) > 0:
            s_last = s_list[-1]
            y_last = y_list[-1]
            ys = float((y_last.T @ s_last))
            yy = float((y_last.T @ y_last))
            H0 = ys / yy if yy != 0 else 1.0
        else:
            H0 = 1.0
        r = H0 * q
        alpha_rev = list(reversed(alpha))
        for i in range(len(s_list)):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            a = alpha_rev[i]
            b = float(rho * (y.T @ r))
            r = r + s * (a - b)
        p = -r
        if self.verbose:
            print(f"[L-BFGS dos-bucles] memoria={len(s_list)} H0={H0:.6g} | norma_gradiente={float(np.linalg.norm(grad)):.6g}")
        return p

    def entrenar(self, X: np.array, y: np.array = None, armijo: bool = False):
        # Añadir bias
        self.X = np.column_stack((np.ones((X.shape[0], 1)), X))
        Xb = self.X
        self.y = y.reshape(-1, 1)
        yb = self.y

        n_params = Xb.shape[1]
        w_curr = np.random.randn(n_params, 1) * 0.01

        s_list = []
        y_list = []
        rho_list = []

        g_curr = self.f_gradiente(Xb, w_curr, yb)
        f_curr = self.f_costo(Xb, yb, w_curr)

        if self.verbose:
            print("=== Entrenamiento L-BFGS ===")
            print(f"[info] X shape: {Xb.shape}, y shape: {yb.shape}, n_params: {n_params}")
            print(f"[info] Inicial: costo={f_curr:.6f}, norma_gradiente={float(np.linalg.norm(g_curr)):.6e}")

        t = 0
        while t <= self.max_iter:
            grad_norm = float(np.linalg.norm(g_curr))
            if self.verbose and (t % 100 == 0):
                print(f"iteración {t} | costo: {f_curr:.6f} | norma_gradiente: {grad_norm:.6e} | memoria={len(s_list)}")
            if grad_norm <= self.epsilon:
                if self.verbose:
                    print("Convergencia por norma del gradiente alcanzada.")
                break

            # Dirección
            if len(s_list) == 0:
                p = -g_curr
            else:
                p = self._two_loop_recursion(g_curr, s_list, y_list, rho_list)

            # Line search backtracking Armijo (si armijo True se aplica condición; si no usamos eta fijo)
            step = float(self.eta if self.eta is not None else 1.0)
            rho_ls = 0.5
            c = 1e-4
            gTp = float((g_curr.T @ p))
            if gTp >= 0:
                p = -g_curr
                gTp = float((g_curr.T @ p))

            back = 0
            max_back = 50
            while back < max_back:
                w_next = w_curr + step * p
                f_next = self.f_costo(Xb, yb, w_next)
                if (not armijo) or (f_next <= f_curr + c * step * gTp):
                    break
                step *= rho_ls
                back += 1
                if step < 1e-16:
                    break

            w_next = w_curr + step * p
            g_next = self.f_gradiente(Xb, w_next, yb)

            s = w_next - w_curr
            yv = g_next - g_curr
            sy = float((s.T @ yv))
            if self.verbose:
                print(f"iteración {t} -> paso={step:.3g} backtracks={back} | sy={sy:.6g}")

            if sy > 1e-12:
                if len(s_list) == self.m:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)
                s_list.append(s); y_list.append(yv); rho_list.append(1.0 / sy)

            w_curr = w_next
            g_curr = g_next
            f_curr = self.f_costo(Xb, yb, w_curr)
            t += 1

        self.w = w_curr
        self.pesos = self.w
        self.costo_final = float(self.f_costo(Xb, yb, self.w))
        if self.verbose:
            print(f"Entrenamiento finalizado en iter {t} | costo final: {self.costo_final:.6f}")
        return w_curr.ravel()

    def proba(self, X):
        Xb = np.column_stack((np.ones((X.shape[0], 1)), X))
        return self.sigmoide(Xb, self.w)

    def predict(self, X):
        Xb = np.column_stack((np.ones((X.shape[0], 1)), X))
        sigm = self.sigmoide(Xb, self.w)
        return (sigm >= 0.5).astype(int).flatten()