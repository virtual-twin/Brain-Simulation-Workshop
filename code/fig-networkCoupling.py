from IPython.display import HTML
from tvbo import Network
from tvbo import SimulationExperiment, Dynamics, Coupling
from tvbo.datamodel.schema import Distribution, Range

from tvbo import SimulationExperiment

exp = SimulationExperiment.from_string('''
label: Kuramoto Synchronization

dynamics:
  name: Kuramoto
  parameters:
    omega:
      value: 1.0
  coupling_inputs:
    I:
      dimension: 1
  state_variables:
    theta:
      equation:
        lhs: Derivative(theta, t)
        rhs: omega + I
      distribution:
        name: Uniform
        domain:
          lo: 0.0
          hi: 6.283185307179586
        seed: 42
  derived_variables:
    activity:
      name: activity
      equation:
        lhs: activity
        rhs: sin(theta)
      record: true

network:
  nodes:
    - id: 0
      label: V1
    - id: 1
      label: V2
    - id: 2
      label: MT
  edges:
    - source: 0
      target: 1
      parameters:
        weight: {value: 0.7}
    - source: 1
      target: 2
      parameters:
        weight: {value: 0.4}
    - source: 0
      target: 2
      parameters:
        weight: {value: 0.2}
  coupling:
    I:
      delayed: false
      incoming_states: [theta]
      local_states: [theta]
      parameters:
        K: {value: 0.4}
      pre_expression:
        rhs: sin(theta_j - theta_i)
      post_expression:
        rhs: K * gx / 3

integration:
  method: Heun
  step_size: 0.01
  duration: 100
''')

result = exp.run('tvboptim')

ani = result.sel(variable='activity').animate()
ani.save('/Users/leonmartin_bih/projects/TVB-O/tvb-ontology-optim-workshop/img/videos/NetworkDynamics/coupled_kuramoto.gif')

exp.network.coupling['I'].parameters['K'].value = 0
result_uncoupled = exp.run('tvboptim')
ani_uncoupled = result_uncoupled.sel(variable='activity').animate()
ani_uncoupled.save('/Users/leonmartin_bih/projects/TVB-O/tvb-ontology-optim-workshop/img/videos/NetworkDynamics/uncoupled_kuramoto.gif')