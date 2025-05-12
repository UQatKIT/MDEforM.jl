# Cost Functionals for the MDE

In its most general form, the cost functional of the MDE is given by
```math
\begin{aligned}
    \Delta_T(\vartheta, X_\epsilon) := \int_{\R^d} \left| \frac1T \int_0^T \exp\left(i u^\top X_\epsilon(t)\right) \, dt - \int_{\R^d} \exp\left(i u^\top x\right) \mu(\vartheta, x) \, dx \right|^2 \varphi(u) \, du.
\end{aligned}
```
Choosing a centered Gaussian density with covariance matrix ``\beta^2 I_d, \beta > 0``, for ``\varphi`` reduces the above formula to
```math
\begin{aligned}
    \Delta_T(\vartheta, X_\epsilon) = 
    &\frac{1}{T^2} \int_0^T \int_0^T k(X_\epsilon(t)-X_\epsilon(s)) \, dt \, ds \\[0.25cm]
    &- \frac{2}{T} \int_0^T (\mu(\vartheta) \ast k)(X_\epsilon(t)) \, dt + \int_{\R^d} (\mu(\vartheta) \ast k)(x) \mu(\vartheta, x) \, dx,
\end{aligned}
```
where ``k(x) :=\exp(-\beta^2 |x|_2^2/2)`` is the characteristic function of ``\varphi`` and ``\ast`` denotes the convolution operator on ``\R^d.`` Furthermore, if ``\mu(\vartheta)`` is the density of a centered multivariate normal distribution with a positive definite covariance matrix ``M(\vartheta) \in \R^{d \times d}``, then
```math
\begin{aligned}
  \Delta_T(\vartheta, X_\epsilon) &= - \frac{2}{T \sqrt{\det(I_d + \beta^2 M(\vartheta)) }} \int_0^T \exp\left( -\frac{\beta^2}{2} X_\epsilon(t)^\top \left( I_d + \beta^2 M(\vartheta) \right)^{-1} X_\epsilon(t) \right) \, dt \\[0.25cm]
    &+ \frac{1}{\sqrt{\det(I_d + 2 \beta^2 M(\vartheta))}}.
\end{aligned}
```

## Cost Functionals

```@docs
k
Δ
```

## Gradients of Cost Functionals

```@docs
Δ_grad_ϑ
Δ_grad_Σ
```

## Index

```@index
Pages = ["MDE_functionals.md", "MDE_gradients.md"]
```
