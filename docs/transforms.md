# Explanation of transforms

Normalization of data for machine learning  is usually very trivial: If, e.g., one scales regression data by a constant or applies a logarithm,
one just needs to apply the inverse operation to the output of the neural network. However, for this application, the normalization
is a bit more complex.

As is included in the code more broadly, one may, e.g., train a second order ODE only by using the loss w.r.t. the state itself.
In fact, in this case (if the initial value is inferred), one should not be restricted to have data available for the derivative
of the state. However, when predicting with the model, the derivative is also predicted and one might be interested in it.

Now, clearly the transformation of the state $x$ and time $t$ will induce a transformation of the derivative $\dot x=\frac{dx}{dt}$.
(The neural ode will integrate $\frac{d}{dt}x=\dot x$ to get $x$, which, when computing the loss, is compared to the transformed data, such that $\dot x$ is also transformed.)

As a Consequence of this, one needs to infer the transformation of the derivative, form the transformation of the state and time. This can be done using the chain rule.
There are some other requirements the transformations need to fulfill:
- Allow general transformations (e.g. $x'=x/\cos(t)$)
- Time must be transformed independently of the state (when predicting, the state is not known yet for the desired time steps to be integrated)
- Not just infer inverse transformation of $\dot x$, but also "forward" transformation
    - Needed when computing helmholtz metrics ore directly evaluating the ode to get $\frac{d}{dt}\dot x$, the second derivative
- Infer the inverse transformation of $\frac{d}{dt}\dot x$

For this reason this package comes with custom `Normalizer`'s that do all of this automatically, once the transformation is supplied.

Of course, all of the above equally applies if only the derivative is given and the state is inferred. First however for the case,
where the state is given and the derivative is inferred.

## Inferring from a state transformation

Let the transformed quantities be denoted by $t',x',\dot x'$. To satisfy the requirements above, let the transformation be given by
$$
\begin{align}
t &= g(t'), &t'&=g^{-1}(t)\\
x &= f(t,x'), &x'&=f^{-1}(t,x)\\
\end{align}
$$
Then, the derivative is (inverse) transformed as (using the chain rule)
$$
\begin{equation}
\dot x=\frac{df}{dt}(t,x')=\frac{\partial f}{\partial t}(t,x')+\frac{\partial f}{\partial x'}(t,x')\frac{dx'}{dt}(t,x')=\frac{\partial f}{\partial t}(t,x')+\frac{\partial f}{\partial x'}(t,x')\frac{dx'}{dt'}(t')\frac{dt'}{dt}(t),
\end{equation}
$$
where $\frac{dt'}{dt}(t) = (\frac{dg}{dt'}(t'))^{-1}$, by the inverse function theorem and $\frac{dx'}{dt'}(t') = \dot x'$, the transformed derivative.

The same calculation can be done for the "forward" transformation of $\dot x'$, but it is not needed for the current implementation, as one can just algebraically invert the above.

To (inverse) transform the second derivative one gets
$$
\begin{align*}
\frac{d}{dt}\frac{dx}{dt} &= \frac{d}{dt}\left(\frac{\partial f}{\partial t} + \frac{\partial f}{\partial x'}\frac{dx'}{dt}\right)\\
&= \frac{\partial^2 f}{\partial t^2} + \frac{\partial^2 f}{\partial x'\partial t}\frac{dx'}{dt}\\
&\phantom{=}+ \frac{\partial^2 f}{\partial t\partial x'}\frac{dx'}{dt} + \frac{\partial^2 f}{\partial x'^2}\left(\frac{dx'}{dt}\right)^2\\
&\phantom{=}+ \frac{\partial f}{\partial x'}\frac{d^2x'}{dt^2}\\
&= \frac{\partial^2 f}{\partial t^2} + \frac{\partial^2 f}{\partial x'\partial t}\frac{dx'}{dt'}\frac{dt'}{dt}\\
&\phantom{=}+ \frac{\partial^2 f}{\partial t\partial x'}\frac{dx'}{dt'}\frac{dt'}{dt} + \frac{\partial^2 f}{\partial x'^2}\left(\frac{dx'}{dt'}\right)^2\left(\frac{dt'}{dt}\right)^2\\
&\phantom{=}+ \frac{\partial f}{\partial x'}\left(\frac{d^2x'}{dt'^2}\left(\frac{dt'}{dt}\right)^2 + \frac{dx'}{dt'}\frac{d^2t'}{dt^2}\right),\\
\end{align*}
$$
where $\frac{d^2t'}{dt^2} = -\frac{d^2g}{dt'^2}(\frac{dg}{dt'})^{-3}$ (see [here](https://en.wikipedia.org/wiki/Inverse_function_rule#Higher_derivatives)) and $\frac{d^2x'}{dt'^2} = \ddot x'$, the transformed second derivative.

Again, this may be algebraically inverted.

## Inferring from a complete transformation

In case of a a complete transformation $x=j(t,x',\dot x'), \dot x = k(t,x',\dot x'), t = g(t')$, where $\dot x$ is present in the training data, a transformation of $\dot x$  is already given. However, the transformation of the second derivative needs to be inferred.

This can again be done using the chain rule. The (inverse) transformation of the second derivative is then given by
$$
\begin{align*}
\ddot x &= \frac{d}{dt}\dot x = \frac{\partial k}{\partial t} + \frac{\partial k}{\partial x'}\frac{dx'}{dt} + \frac{\partial k}{\partial \dot x'}\frac{d\dot x'}{dt}\\
&= \frac{\partial k}{\partial t} + \frac{\partial k}{\partial x'}\frac{dx'}{dt'}\frac{dt'}{dt} + \frac{\partial k}{\partial \dot x'}\frac{d\dot x'}{dt'}\frac{dt'}{dt},\\
\end{align*}
$$
where $\frac{d\dot x'}{dt'} = \ddot x'$, the transformed second derivative.
