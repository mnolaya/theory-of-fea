# Rotation

$$
R = \begin{bmatrix}
cos(\theta) & sin(\theta) \\
-sin(\theta) & cos(\theta)
\end{bmatrix}
$$

$$
\sigma_{loc} = R \cdot \sigma_{glob} \cdot R^{T}
$$

Explicitly...

$$
\begin{align*}
\sigma_{11} &= \sigma_{xx}cos^{2}(\theta) + \sigma_{yy}sin^{2}(\theta) + 2\sigma_{xy}sin(\theta)cos(\theta) \\
\sigma_{22} &= \sigma_{xx}sin^{2}(\theta) + \sigma_{yy}cos^{2}(\theta) - 2\sigma_{xy}sin(\theta)cos(\theta) \\
\sigma_{12} &= (\sigma_{yy} - \sigma_{xx})sin(\theta)cos(\theta) + \sigma_{xy}(cos^{2}(\theta) - sin^{2}(\theta))
\end{align*}
$$

Strain terms

$$
\begin{align*}
\varepsilon_{11} &= \varepsilon_{yy}sin^{2}(\theta) \\
\varepsilon_{22} &= \varepsilon_{yy}cos^{2}(\theta) \\
\varepsilon_{12} &= \varepsilon_{yy}sin(\theta)cos(\theta)
\end{align*}
$$


## $G_{12}$

Unknowns:

$G_{12}, \sigma_{xx}, \sigma_{xy}$

### $\theta = 30$

$$
\sin(30) = 1/2 \\
\cos(30) = \sqrt(3)/2
$$

$$
\begin{align*}
\varepsilon_{12} = \frac{\sqrt(3)}{4}\varepsilon_{yy} &= \frac{1}{2G_{12}}((\sigma_{yy} - \sigma_{xx})\frac{\sqrt(3)}{4} + \sigma_{xy}(\frac{3}{4} - \frac{1}{4})) \\
\frac{\sqrt(3)\delta}{40} &= \frac{1}{2G_{12}}(\frac{\sqrt(3)F}{8t} - \frac{\sqrt(3)\sigma_{xx}}{4} + \frac{\sigma_{xy}}{2}) \\
\frac{\sqrt(3)\delta G_{12}}{20} &= \frac{\sqrt(3)F}{8t} - \frac{\sqrt(3)\sigma_{xx}}{4} + \frac{\sigma_{xy}}{2} \\
\end{align*}
$$

$$
\therefore \frac{\sqrt(3)\sigma_{xx}}{4} + \frac{\sqrt(3)G_{12}\delta}{20} - \frac{\sigma_{xy}}{2} = \frac{\sqrt(3)F}{8t}
$$

### $\theta = 45$

$$
\sin(30) = \sqrt(2)/2 \\
\cos(30) = \sqrt(2)/2
$$

$$
\begin{align*}
\varepsilon_{12} = \frac{1}{2}\varepsilon_{yy} &= \frac{1}{2G_{12}}((\sigma_{yy} - \sigma_{xx})\frac{1}{2}) \\
\frac{\delta}{20} &= \frac{1}{2G_{12}}(\frac{F}{4t} - \frac{\sigma_{xx}}{2}) \\
\frac{G_{12}\delta}{10} &= (\frac{F}{4t} - \frac{\sigma_{xx}}{2})
\end{align*}
$$

$$
\therefore \frac{\sigma_{xx}}{2} + \frac{G_{12}\delta}{10} = \frac{F}{4t}
$$

### $\theta = 60$

$$
\sin(30) = \sqrt(3)/2 \\
\cos(30) = 1/2
$$

$$
\begin{align*}
\varepsilon_{12} = \frac{\sqrt(3)}{4}\varepsilon_{yy} &= \frac{1}{2G_{12}}((\sigma_{yy} - \sigma_{xx})\frac{\sqrt(3)}{4} + \sigma_{xy}(\frac{1}{4} - \frac{3}{4})) \\
\frac{\sqrt(3)\delta}{40} &= \frac{1}{2G_{12}}(\frac{\sqrt(3)F}{8t} - \frac{\sqrt(3)\sigma_{xx}}{4} - \frac{\sigma_{xy}}{2}) \\
\frac{\sqrt(3)\delta G_{12}}{20} &= \frac{\sqrt(3)F}{8t} - \frac{\sqrt(3)\sigma_{xx}}{4} - \frac{\sigma_{xy}}{2} \\
\end{align*}
$$

$$
\therefore \frac{\sqrt(3)\sigma_{xx}}{4} + \frac{\sqrt(3)G_{12}\delta}{20} + \frac{\sigma_{xy}}{2} = \frac{\sqrt(3)F}{8t}
$$

3 equations, 3 unknowns

$$
q = \begin{bmatrix}
G_{12} \\
\sigma_{xx} \\
\sigma_{xy} \\
\end{bmatrix}
$$

$$
K = \begin{bmatrix}
\sqrt(3)\delta/{20} & \sqrt(3)/{4} & -1/2 \\
\delta/{10} & 1/{2} & 0 \\
\sqrt(3)\delta/{20} & \sqrt(3)/{4} & 1/2 \\
\end{bmatrix}
$$

$$
F = \begin{bmatrix}
\sqrt(3)F/8t \\
F/4t \\
\sqrt(3)F/8t \\
\end{bmatrix}
$$


