# 3. Dynamical perspective

## How to analyse a brain system

The key idea for this section is that a dynamical system is not only a
simulator. It is a geometry. We can look at the same model from three angles,
and each angle answers a different question.

Time series answer: what does the model do after I start it somewhere? This is
the most direct link to data. We simulate $x(t)$ at fixed parameters and compare
the resulting signal with EEG, MEG, fMRI, spectra, functional connectivity, or
FCD.

The phase plane answers: how does the model move through its state space? For a
two-dimensional system, every point in the plane has an arrow attached to it.
That arrow says where the system wants to go next. Fixed points are places where
the arrows vanish. Nullclines are curves where one component of the motion is
zero. Trajectories show which regions are attracted to which fixed point or
cycle. This is a local geometric view at one parameter value.

The bifurcation diagram answers: when does the qualitative behaviour change?
Instead of fixing the parameter, we vary it and track equilibria and periodic
orbits. This gives a map of regimes: resting states, oscillatory states,
multistability, folds, Hopf points and abrupt transitions. In brain modelling,
this is often the most useful view because we care about transitions between
healthy, oscillatory, seizure-like, or pathological regimes.

The intuitive sentence for the slide is: time series are the movie, the phase
plane is the landscape, and the bifurcation diagram is the atlas of all
landscapes as a parameter changes.

## Linear stability

Start with the smallest possible example:

$$
\dot x = a x,
\qquad
x(t)=x_0 e^{at}.
$$

There is one fixed point at $x=0$. Whether it attracts or repels depends only on
the sign of $a$.

This is where the symbol $\lambda$ enters. In stability analysis, $\lambda$ is
an eigenvalue of the local linearised system. Intuitively, it is the growth rate
of a tiny perturbation away from a fixed point. If $\lambda<0$, the perturbation
shrinks and the fixed point attracts nearby states. If $\lambda>0$, the
perturbation grows and the fixed point repels nearby states. If $\lambda=0$, the
linear approximation is exactly neutral: this is the boundary where stability
can change.

In this one-dimensional example, the Jacobian is just the number $a$, so the
single eigenvalue is

$$
\lambda=a.
$$

That is why the plot labels the dashed neutral trajectory as $a=0$,
$\lambda=0$. The line itself is not decaying and not growing. It marks the
change point between decay and growth.

If $a<0$, then $e^{at}$ decays and the state returns to zero. The fixed point is
stable. If $a>0$, then $e^{at}$ grows and any small perturbation moves away from
zero. The fixed point is unstable. If $a=0$, the system is exactly at the
borderline. This is the simplest possible bifurcation point.

For a nonlinear system $\dot x=f(x)$, we use the same idea locally. Near a fixed
point $x^\ast$, write $x=x^\ast+\delta x$ and linearise:

$$
\dot{\delta x} = J\delta x,
\qquad
J = \left.\frac{\partial f}{\partial x}\right|_{x^\ast}.
$$

In many dimensions, stability is read from the eigenvalues of $J$. If every
eigenvalue has negative real part, perturbations decay. If one eigenvalue has
positive real part, perturbations grow. A bifurcation occurs when an eigenvalue
crosses the imaginary axis as a parameter changes.

Real eigenvalue crossing zero gives saddle-node, pitchfork or transcritical
bifurcations. A complex conjugate pair crossing the imaginary axis gives a Hopf
bifurcation, which is the birth or death of an oscillation.

## 2D portraits

The phase portraits are meant to make stability visible without reading a long
equation. The arrows show the vector field. The black trajectories show example
solutions. The dots mark fixed points.

For a stable node, trajectories approach the fixed point without rotation. For a
saddle, one direction attracts and another direction repels. This creates a
separatrix: a boundary between qualitatively different futures. For a stable
focus, trajectories spiral into the fixed point. For a centre, trajectories
circle around without damping, so the amplitude is conserved.

The important brain-modelling intuition is that local geometry can already tell
us what kind of signal we should expect. A node gives relaxation. A focus gives
damped oscillations. A centre or limit cycle gives sustained oscillations. A
saddle can create thresholds and switches.

## Regime-change building blocks

The normal forms are the canonical local pictures. They are not meant to be
full brain models. They are the small mathematical templates that appear near
many different transitions.

The slide now shows each example in two linked views. The top row is the
bifurcation diagram: equilibria as the parameter $a$ changes. The dotted
vertical line marks one chosen parameter value. The bottom row shows the scalar
vector field at exactly that value: $\dot x=f(x)$ as a function of $x$. Where
the curve crosses zero, the system has a fixed point. Filled dots are stable
fixed points; open dots are unstable fixed points. The arrows show the local
flow direction along the state axis.

Saddle-node:

$$
\dot x = a - x^2.
$$

For one side of the parameter there are two fixed points, one stable and one
unstable. At the fold, they collide and disappear. This is the local geometry
of an abrupt regime shift. In brain models, this is the kind of mechanism that
can underlie sudden onset or offset transitions.

Pitchfork:

$$
\dot x = a x - x^3.
$$

For $a<0$, the symmetric state $x=0$ is stable. For $a>0$, that symmetric state
becomes unstable and two new stable states appear. This is symmetry breaking.
The system must choose one of two equivalent states. The important visual cue is
that the central branch continues, but its stability changes at the branching
point.

Hysteresis:

$$
\dot x = a + x - x^3.
$$

This is the unfolded, tilted version of the pitchfork. The symmetry is broken,
and instead of a clean fork we get an S-shaped curve. In the middle parameter
range, two stable states coexist. Which one the system occupies depends on its
history. This is why hysteresis is a natural mathematical image for memory,
switching and delayed recovery.

The message for the audience: these pictures are universal local geometries.
Different detailed models can have the same local transition because near the
bifurcation they reduce to the same normal form.

## Hopf: birth of an oscillation

The Hopf normal form is easiest to understand in the complex plane:

$$
\dot z = (a+i\omega)z - |z|^2z,
\qquad z=x_1+i x_2.
$$

When $a<0$, the origin is stable. Trajectories spiral inward. When $a>0$, the
origin becomes unstable, but the cubic term prevents indefinite growth. The
system settles on a stable circle with radius

$$
r=\sqrt a.
$$

That circle is a stable limit cycle. In the time series, it appears as a
sustained oscillation. In the phase plane, it appears as a closed orbit. In the
bifurcation diagram, it appears as a branch of periodic orbits born at the Hopf
point.

This is the bridge to brain rhythms: a Hopf bifurcation is the cleanest local
mechanism by which a resting fixed point can turn into a self-sustained
oscillation.

## Numerical continuation

Simulation asks: what happens from this initial condition? Continuation asks:
where are all the equilibria and cycles, and how do they change as a parameter
moves?

We solve

$$
F(x,a)=0
$$

for equilibria, but we do not simply step along the parameter $a$. At a fold,
the branch turns around, so $x$ is no longer a single-valued function of $a$.
Pseudo-arclength continuation tracks the branch by an internal path variable
$s$:

$$
(x,a)=(x(s),a(s)).
$$

Each step has two parts. The predictor moves a short distance along the tangent
of the branch. The corrector uses Newton's method on an augmented system to land
back on $F(x,a)=0$. Because the step follows the branch geometry rather than the
parameter axis, it can continue through folds.

Along the way, the algorithm monitors test functions. A zero eigenvalue marks a
fold or related real-eigenvalue bifurcation. A complex pair crossing the
imaginary axis marks a Hopf bifurcation. From a Hopf point, branch switching can
start the continuation of periodic orbits.

In TVBO, the user describes this as a `Continuation` object. The backend
`bifurcationkit.jl` performs the numerical continuation, and the result can be
plotted directly:

```python
result = exp.run("bifurcationkit.jl")
result.continuations["my_cont"].plot_3d(VOI="x1")
```

## Generic2dOscillator: regimes over input I

The Generic2dOscillator example is the transition from textbook normal forms to
a brain-relevant neural mass model. Here the swept parameter is the input
current $I$. The bifurcation diagram shows where the model rests, where it
oscillates, and where abrupt transitions occur.

Stable fixed-point branches correspond to resting or persistent-activity
regimes. Periodic-orbit branches correspond to oscillatory regimes. Fold points
mark abrupt jumps. Hopf points mark where oscillations are born or disappear.

The panel with time series is important because it reconnects the abstract
diagram to observable signals. A point on a fixed branch gives a steady or
relaxing signal. A point on a periodic branch gives a rhythm. This is the main
lesson: continuation is not an alternative to simulation. It organises many
simulations into one regime map.

## 3.1 Bifurcation analysis: hands-on

In the hands-on notebook, the audience should keep three questions in mind:

1. Which parameter controls the transition?
2. Which branches are stable, unstable or periodic?
3. What signal would I observe if I simulated the system at a chosen point in
   the diagram?

The practical goal is to run the same `Continuation` schema on the
Generic2dOscillator, sweep $I$, continue fixed points and periodic orbits, then
read the output as a map of resting, oscillatory and transition regimes.

## TVB simulation infused with bifurcation analysis

This final dynamical slide overlays simulation on top of continuation. The
background is the Generic2dOscillator bifurcation diagram in the input parameter
$I$. The moving traces are nodes of a Desikan-Killiany network simulation,
generated by TVBO and run through the tvboptim backend. For each node we plot its current
voltage $V_i(t)$ against its effective input $I_{\mathrm{eff},i}(t)$, which
combines the current drive level with structural network input from connected
neighbours.

The point is that a network simulation is not just a collection of time series.
At every moment, each node occupies a position in the regime map. Nodes with low
effective input sit near resting fixed-point regions. Nodes whose effective
input crosses the oscillatory window move into the part of the diagram where
periodic orbits exist. Coupling makes the nodes move through this map together.
The last part of the animation holds the drive fixed so the audience can see
where the network states settle. The bifurcation diagram becomes a coordinate
system for reading the network dynamics.

The speaker line is: the bifurcation diagram is the atlas, and the network
simulation is a moving path through that atlas.