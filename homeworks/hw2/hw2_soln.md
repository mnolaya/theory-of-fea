# Homework 2 Solution

Michael N. Olaya

July 2024

> [!Note]
> Coordinate system notation used in this homework is:
> $$
> (x, y) \rightarrow (x_{1}, x_{2}) \\
> (\xi, \eta) \rightarrow (\eta_{1}, \eta_{2})
> $$

## Problem 1

### 4-noded 2D element (linear shape functions)

**Node 1**

$$
N_{1} = (\eta_{1} - 1)(\eta_{2} - 1)*C
$$

@ $\mathbf{\eta} = (-1, -1)$

$$
N_{1} = 1 = (-1 - 1)(-1 - 1)*C \\
\rightarrow 1 = (-2)(-2)*C \rightarrow C = 1/4 \\
$$

$$
\therefore N_{1} = \frac{1}{4}(\eta_{1} - 1)(\eta_{2} - 1)
$$

---

**Node 2**

$$
N_{2} = (\eta_{1} + 1)(\eta_{2} - 1)*C
$$

@ $\mathbf{\eta} = (1, -1)$

$$
N_{2} = 1 = (1 + 1)(-1 - 1)*C \\
\rightarrow 1 = (2)(-2)*C \rightarrow C = -1/4 \\
$$

$$
\therefore N_{2} = -\frac{1}{4}(\eta_{1} + 1)(\eta_{2} - 1)
$$

---

**Node 3**

$$
N_{3} = (\eta_{1} + 1)(\eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (1, 1)$

$$
N_{3} = 1 = (1 + 1)(1 + 1)*C \\
1 = (2)(2)*C \rightarrow C = 1/4 \\
$$

$$
\therefore N_{3} = \frac{1}{4}(\eta_{1} + 1)(\eta_{2} + 1)
$$

---

**Node 4**

$$
N_{4} = (\eta_{1} - 1)(\eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (-1, 1)$

$$
N_{4} = 1 = (-1 - 1)(1 + 1)*C \\
\rightarrow 1 = (-2)(2)*C \rightarrow C = -1/4 \\
$$

$$
\therefore N_{4} = -\frac{1}{4}(\eta_{1} - 1)(\eta_{2} + 1)
$$

### 8-noded 2D element (quadratic shape functions)

**Node 1**

$$
N_{1} = (\eta_{1} - 1)(\eta_{2} - 1)(\eta_{1} + \eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (-1, -1)$

$$
N_{1} = 1 = (-1 - 1)(-1 - 1)(-1 + -1 + 1)*C \\
\rightarrow 1 = (-2)(-2)(-1)*C \rightarrow C = -1/4 \\
$$

$$
\therefore N_{1} = -\frac{1}{4}(\eta_{1} - 1)(\eta_{2} - 1)(\eta_{1} + \eta_{2} + 1)
$$

---

**Node 2**

$$
N_{2} = (\eta_{1} - 1)(\eta_{1} + 1)(\eta_{2} - 1)*C
$$

@ $\mathbf{\eta} = (0, -1)$

$$
N_{2} = 1 = (0 - 1)(0 + 1)(-1 - 1)*C \\
\rightarrow 1 = (-1)(1)(-2)*C \rightarrow C = 1/2 \\
$$

$$
\therefore N_{1} = \frac{1}{2}(\eta_{1} - 1)(\eta_{1} + 1)(\eta_{2} - 1)
$$

---

**Node 3**

$$
N_{3} = (\eta_{1} + 1)(\eta_{2} - 1)(\eta_{1} - \eta_{2} - 1)*C
$$

@ $\mathbf{\eta} = (1, -1)$

$$
N_{3} = 1 = (1 + 1)(-1 - 1)(1 - (-1) - 1)*C \\
\rightarrow 1 = (2)(-2)(1)*C \rightarrow C = -1/4 \\
$$

$$
\therefore N_{1} = -\frac{1}{4}(\eta_{1} + 1)(\eta_{2} - 1)(\eta_{1} - \eta_{2} - 1)
$$

---

**Node 4**

$$
N_{4} = (\eta_{1} + 1)(\eta_{2} - 1)(\eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (1, 0)$

$$
N_{4} = 1 = (1 + 1)(0 - 1)(0 + 1)*C \\
\rightarrow 1 = (2)(-1)(1)*C \rightarrow C = -1/2 \\
$$

$$
\therefore N_{4} = -\frac{1}{2}(\eta_{1} + 1)(\eta_{2} - 1)(\eta_{2} + 1)
$$

---

**Node 5**

$$
N_{5} = (\eta_{1} + 1)(\eta_{2} + 1)(\eta_{2} - \eta_{1} + 1)*C
$$

@ $\mathbf{\eta} = (1, 1)$

$$
N_{5} = 1 = (1 + 1)(1 + 1)(1 - 1 + 1)*C \\
\rightarrow 1 = (2)(2)(1)*C \rightarrow C = 1/4 \\
$$

$$
\therefore N_{5} = \frac{1}{4}(\eta_{1} + 1)(\eta_{2} + 1)(\eta_{2} - \eta_{1} + 1)
$$

---

**Node 6**

$$
N_{6} = (\eta_{1} - 1)(\eta_{1} + 1)(\eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (0, 1)$

$$
N_{6} = 1 = (0 - 1)(0 + 1)(1 + 1)*C \\
\rightarrow 1 = (-1)(1)(2)*C \rightarrow C = -1/2 \\
$$

$$
\therefore N_{6} = -\frac{1}{2}(\eta_{1} - 1)(\eta_{1} + 1)(\eta_{2} + 1)
$$

---

**Node 7**

$$
N_{7} = (\eta_{1} - 1)(\eta_{2} + 1)(\eta_{1} - \eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (-1, 1)$

$$
N_{7} = 1 = (-1 - 1)(1 + 1)(-1 - 1 + 1)*C \\
\rightarrow 1 = (-2)(2)(-1)*C \rightarrow C = 1/4 \\
$$

$$
\therefore N_{7} = \frac{1}{4}(\eta_{1} - 1)(\eta_{2} + 1)(\eta_{1} - \eta_{2} + 1)
$$

---

**Node 8**

$$
N_{8} = (\eta_{1} - 1)(\eta_{2} - 1)(\eta_{2} + 1)*C
$$

@ $\mathbf{\eta} = (-1, 0)$

$$
N_{8} = 1 = (-1 - 1)(0 - 1)(0 + 1)*C \\
\rightarrow 1 = (-2)(-1)(1)*C \rightarrow C = 1/2 \\
$$

$$
\therefore N_{8} = \frac{1}{2}(\eta_{1} - 1)(\eta_{2} - 1)(\eta_{2} + 1)
$$

## Problem 2