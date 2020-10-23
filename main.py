import libpymath as lpm
import network
import matplotlib.pyplot as plt
import math
import random

# Just a matrix storing 1
ONE = lpm.matrix.Matrix(1, 1)
ONE[0, 0] = 1

# A matrix storing 0
ZERO = lpm.matrix.Matrix(1, 1)
ZERO[0, 0] = 0


# Generate a circle with some standard deviation (std)
def generateNoise(points, std=0.1):
    res = []
    theta = 0
    inc = math.pi * 2 / points

    while theta < math.pi * 2:
        res.append([math.cos(theta) + random.uniform(-std / 2, std / 2), math.sin(theta) + random.uniform(-std / 2, std / 2)])
        theta += inc

    return res[:points]


# The training data
data = []

# Generate training data
for i in range(1000):
    points = generateNoise(100, std=0 if i != 0 else 0.5)
    x = []
    y = []

    for p in points:
        x.append(p[0])
        y.append(p[1])

    data.append(x + y)

# The generator network
generator = network.Network(layers=(5, 70, 200), lr=0.02, activations=[lpm.network.TANH] * 2)
# The discriminator network
discriminator = network.Network(layers=(200, 100, 25, 1), lr=0.02, activations=[lpm.network.TANH] * 3)

# Log the loss for both networks
discriminator.log("loss", 1)
generator.log("loss", 1)


# ============================================================================================ #
# =================== Actual training of the GAN and testing it's output # =================== #
# ============================================================================================ #

# Train the GAN. NOT WORKING....
def trainGAN(gen, disc, image):
    # Generate the generator input noise
    noise = lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)], rows=5)

    # Get the discriminator output from the generator
    generated = gen.feedForward(noise)

    # Train the discriminator on a dataset image -- We want the discriminator to output 1 (i.e. is a dataset image)
    disc.backpropagate(image, ONE)

    # Train the discriminator on a generated image -- We want the discriminator to output 0 (i.e. is a generated image)
    disc.backpropagate(generated, ZERO)

    # Backpropagate over both networks to train the generator without training the discriminator.
    # The output of this should be 1 as the generator's output through the discriminator should
    # mimic a real dog image

    # network.Network.backpropagateDual(gen, disc, noise, ONE)

    # Using this one as it uses equations to calculate the error perfectly
    # It simply replaces the discriminator
    network.Network.backpropagateNonDual(gen, noise)


# Generate the noise -- same noise for testing purposes
noise = lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)], rows=5)

# Get the untrained generator's output from the generated noise
preTrain = generator.feedForward(noise)

# Train the generator
for i in lpm.progress.Progress(range(10000)):
    # Don't pick the first bit of data -- it is for testing purposes
    index = random.randint(1, len(data) - 1)

    # Train the GAN (sort of)
    trainGAN(generator, discriminator, data[index])

# Get the output of the generator after training (using the equations
postTrain = generator.feedForward(noise)

# Now train the GAN using the discriminator (which has been trained) -- this is for testing purposes to see what happens
for i in lpm.progress.Progress(range(10000)):
    network.Network.backpropagateDual(generator, discriminator, noise, lpm.matrix.Matrix(data=[[1]], rows=1))

# Get the output of the generator after training using the
# discriminator (it *should* be the same / better as the output after training initially)
finalTrain = generator.feedForward(noise)

# Plot some graphs
plt.scatter(data[0][:100], data[0][100:])
plt.scatter(data[1][:100], data[1][100:])
plt.scatter(preTrain.toList()[:100], preTrain.toList()[100:])
plt.scatter(postTrain.toList()[:100], postTrain.toList()[100:])
plt.scatter(finalTrain.toList()[:100], finalTrain.toList()[100:])
plt.show()

# Print some info
print("Discriminator on pre-trained     :", discriminator.feedForward(preTrain))
print("Discriminator on trained         :", discriminator.feedForward(postTrain))
print("Discriminator on dataset (noisy) :", discriminator.feedForward(data[0]))
print("Discriminator on dataset (clean) :", discriminator.feedForward(data[1]))

print("\n\n\n\n\n\n\n")

# This is some testing
network.Network.backpropagateDualNoChange(generator, discriminator, noise, lpm.matrix.Matrix(data=[[1]], rows=1))
print("\n")
network.Network.backpropagateNonDualNoChange(generator, noise)
