# Isoparametric shape functions

## Coordinate systems

- Global $\rightarrow (X_{1}, X_{2}, X_{3})$
- Local (element) $\rightarrow (x_{1}, x_{2}, x_{3})$
- Natural (isoparametric) $\rightarrow (\eta_{1}, \eta_{2}, \eta_{3})$

## Relating coordinate systems in 1D

A domain $\Omega$ can be represented in terms of $\mathbf{x}$ or $\mathbf{\eta}$. For a single element, define the local element domain as

$$
x \in (0, l) \quad | \quad \eta \in (-1, 1)
$$

Then, relate the two coordinate systems with each other:

$$
x_{i} = 0 \rightarrow \eta_{i} = -1
$$

$$
x_{i} = l/2 \rightarrow \eta_{i} = 0
$$

$$
x_{i} = l \rightarrow \eta_{i} = 1
$$

Writing these relationships as functions (e.g., $\eta = f(x)$)

$$
\eta = f(x) = \frac{2}{l}x - 1
$$

$$
x = f(\eta) = \frac{l}{2}(\eta + 1)
$$

### Jacobian

Taking the derivative of the above relationships, we can derive the Jacobian $J$.

$$
\frac{d\eta}{dx} = 2/l = J^{-1}
$$

$$
\frac{dx}{d\eta} = l/2 = J
$$

$$
\therefore \mathbf{J} = \frac{d\mathbf{x}}{d\mathbf{\eta}} \leftarrow \text{vector form}
$$

## Deriving linear shape functions in 1D

For a simple 2 noded 1D element, we want to define two "shape" functions $N_{i}$ that evaluate to $1$ at the corresponding node and are zero at all other nodes. For example, shape function $N_{1}$ should be equivalent to $1$ when $x = 0$ and $0$ at $x = l$. 

**@ $x = 0$**

$$
N_{1} = (x - l)*C \\
N_{1}(x = 0) = 1 = (-l)*C \Rightarrow C = \frac{-1}{l}
$$

$$
\therefore N_{1}(x) = 1 - \frac{x}{l}
$$

Substituting $x = f(\eta)$...

$$
N_{1}(\eta) = \frac{1}{2}(1 - \eta)
$$

**@ $x = l$**

$$
N_{2} = \frac{x}{L}*C \\
N_{2}(x = l) = 1 = C \Rightarrow C = 1
$$

$$
\therefore N_{2}(x) = \frac{x}{l}
$$

Substituting $x = f(\eta)$...

$$
N_{2}(\eta) = \frac{1}{2}(1 + \eta)
$$

> [!IMPORTANT]
> Shape functions are...
> - Differentiable (required for formulating weak form)
> - Continuous across element boundaries
> - $\sum N_{i} = 1$ at all points in the element (i.e., weighted average of nodal values) 

## What does "isoparametric" mean?

For elements to be considered isoparametric, the same shape function used to interpolate displacements are used to interpolate (map) coordinates.

Displacements: $\mathbf{u} = \mathbf{N}\mathbf{q}$

Coordinates: $\mathbf{x} = \mathbf{N}\mathbf{\^{x}}$

In the above, $\mathbf{u}$ are the displacements within the element, $\mathbf{q}$ are the nodal displacements, $\mathbf{x}$ are the coordinates within the element, $\mathbf{\^{x}}$ are the nodal coordinates, and $\mathbf{N}$ are the shape functions.


## Linear shape functions by coordinate system for a 1D, 2-noded element

|Coordinate System|Node 1|Node 2|
|--|--|--|
| Global | $N_{1} = \frac{X_{2} - X}{X_{2} - X_{1}}$ | $N_{2} = \frac{X - X_{1}}{X_{2} - X_{1}}$ |
| Local | $N_{1} = 1 - \frac{x}{l}$ | $N_{2} = \frac{x}{l}$ |
| Natural | $N_{1} = \frac{1}{2}(1 - \eta)$ | $N_{2} = \frac{1}{2}(1 + \eta)$ |

To map from natural to local or global coordinates:

$$
x = N_{1}(\eta)*0 + N_{2}(\eta)*l
$$

$$
X = N_{1}(\eta)*X_{1} + N_{2}(\eta)*X_{2}
$$

## Deriving shape functions in 2D

For a 2D element with $n$ nodes, there are $n*2$ shape functions that need to be derived. As an example, consider an 8-noded element in 2D. The displacements in the element $\mathbf{u}$ (16 x 1) are related to the nodal displacements $\mathbf{q}$ (2 x 1) and the shape functions $\mathbf{N}$ (2 x 16).

$$
\mathbf{q} =
\begin{bmatrix}
q_{11} \\
q_{12} \\
\vdots \\
q_{81} \\
q_{82}
\end{bmatrix}
\quad
\mathbf{u} =
\begin{bmatrix}
u_{1} \\
u_{2}
\end{bmatrix}
\quad
\mathbf{N} =
\begin{bmatrix}
N_{11} & N_{21} & \dots & N_{81} \\
N_{12} & N_{22} & \dots & N_{82}
\end{bmatrix}
\Rightarrow
\mathbf{u} = \mathbf{N}\mathbf{q}
$$

Note that the first index refers to the node number and the second to the coordinate system component. E.g., $q_{72}$ is the displacement at node 7 in the 2-direction.

As with 1D element, the shape functions must be derived such that they evaluate to $1$ at the corresponding node, and $0$ at all other nodes. For a 4-noded 2D element in the natural coordinate system:

$$
N_{1} = (\eta_{1} - 1)(\eta_{2} - 1)*C
$$

@ $\mathbf{\eta} = (-1, 1)$

$$
N_{1} = 1 = (-1 - 1)(-1 - 1)*C \rightarrow C = \frac{1}{4}
$$

$$
\therefore N_{1} = \frac{1}{4}(\eta_{1} - 1)(\eta_{2} - 1)
$$

In a 8-noded 2D element example, nodes exist not only at element vertices, but along the length of the element sides. Therefore, additional terms must be included to enforce the evaluation conditions noted above.

$$
N_{1} = (\eta_{1} - 1)(\eta_{2} - 1)(-\eta_{1} - \eta_{2} - 1)*C
$$

The term $(-\eta_{1} - \eta_{2} - 1)$ is selected by examining the nodes adjacent to $\mathbf{\eta} = (-1, 1)$, which are $\mathbf{\eta} = (-1, 0)$ and $\mathbf{\eta} = (0, -1)$. Linear interpolation between the adjacent nodes yields:

$$
\frac{\eta_{2} - \eta_{12}}{\eta_{1} - \eta_{11}} = \frac{\eta_{22} - \eta_{12}}{\eta_{21} - \eta_{11}} \\
$$

$$
\rightarrow \frac{\eta_{2} - 0}{\eta_{1} - (-1)} = \frac{-1 - 0}{0 - (-1)}
$$

$$
\therefore 0 = -\eta_{2} -\eta_{1} - 1
$$

The interpolation function @ $\mathbf{\eta} = (-1, 1)$ now becomes

$$
N_{1} = 1 = (-1 - 1)(-1 - 1)(-(-1) - (-1) - 1)*C
$$

$$
N_{1} = 1 = (-2)(-2)(1)*C \rightarrow C = \frac{1}{4}
$$

$$
\therefore N_{1} = \frac{1}{4}(\eta_{1} - 1)(\eta_{2} - 1)(-\eta_{1} - \eta_{2} - 1)
$$

