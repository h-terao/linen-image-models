import jax.numpy as jnp
import jax.random as jr
import limo


model = limo.create_model("efficientnet_b0", True)
variables = model.init(jr.PRNGKey(1234), jnp.zeros((224, 224, 3)))

variables = limo.load_pretrained(variables, "efficientnet_b0", True)
model.apply(variables, jnp.zeros((224, 224, 3)))
print("PASS")
