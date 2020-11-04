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
    xPos = []
    yPos = []
    theta = 0
    inc = (math.pi * 2) / points

    while theta < math.pi * 2:
        xPos.append(math.cos(theta) + random.uniform(-std / 2, std / 2))
        yPos.append(math.sin(theta) + random.uniform(-std / 2, std / 2))

        theta += inc

    return xPos[:points] + yPos[:points]


# The training data
data = []

# Generate training data
for i in range(1000):
    data.append(generateNoise(100, std=0 if i != 0 else 0.5))

# The generator network
generator = network.Network(layers=(5, 70, 200), lr=0.02, activations=[lpm.network.TANH] * 2)
# The discriminator network
discriminator = network.Network(layers=(200, 100, 25, 1), lr=0.02, activations=[lpm.network.TANH] * 3)

# Log the loss for both networks
discriminator.log("loss", 1)
generator.log("loss", 1)


# ============================================================================================ #
# =================== Actual training of the GAN and testing it's output ===================== #
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
    network.Network.backpropagateDual(gen, disc, noise, ONE)

if True:
    for i in lpm.progress.Progress(range(10000)):
        op = random.randint(0, 2)

        if op == 0:
            # Generate a perfect circle
            inputData = generateNoise(100, 0)
            output = 1
        elif op == 1:
            # Generate a circle with std=0.1
            inputData = generateNoise(100, random.uniform(0.05, 0.5))
            output = 1
        elif op == 2:
            # Generate random noise
            inputData = [random.uniform(-1, 1) for _ in range(200)]
            output = 0
        else:
            print("The matrix has broken")

            # Purely so that the linting doesn't highlight something
            inputData = []
            output = 0

            exit()

        discriminator.backpropagate(inputData, [output])

print("Discriminator on data (std=0)   (1):", discriminator.feedForward(generateNoise(100, 0)))
print("Discriminator on data (std=0.1) (1):", discriminator.feedForward(generateNoise(100, 0.1)))
print("Discriminator on data (std=0.2) (1):", discriminator.feedForward(generateNoise(100, 0.2)))
print("Discriminator on data (std=0.3) (1):", discriminator.feedForward(generateNoise(100, 0.3)))
print("Discriminator on data (std=0.4) (1):", discriminator.feedForward(generateNoise(100, 0.4)))
print("Discriminator on data (std=0.5) (1):", discriminator.feedForward(generateNoise(100, 0.5)))
print("Discriminator on data (random)  (0):", discriminator.feedForward([random.uniform(-1, 1) for _ in range(200)]))

print("\n=====================================================\n")

print("Getting untrained generator output")
untrained = generator.feedForward([random.uniform(-1, 1) for _ in range(5)])

print("Training the generator")
for i in lpm.progress.Progress(range(10000)):
    # network.Network.backpropagateDual(generator, discriminator, lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)]), lpm.matrix.Matrix(data=[[1]]))
    trainGAN(generator, discriminator, data[random.randint(0, len(data) - 1)])

trained = generator.feedForward([random.uniform(-1, 1) for _ in range(5)])

print("Discriminator on untrained:", discriminator.feedForward(untrained))
print("Discriminator on trained  :", discriminator.feedForward(trained))

plt.scatter(untrained.toList()[:100], untrained.toList()[100:], color="red")
plt.scatter(trained.toList()[:100], untrained.toList()[100:], color="green")
plt.show()
