import libpymath as lpm
import network
import matplotlib.pyplot as plt
import math
import random


# import wandb

# wandb.init(project="simple-gan")


def generateNoise(points, std=0.1):
    res = []
    theta = 0
    inc = math.pi * 2 / points

    while theta < math.pi * 2:
        res.append([math.cos(theta) + random.uniform(-std / 2, std / 2) , math.sin(theta) + random.uniform(-std / 2, std / 2)])
        theta += inc

    return res[:points]


def trainGAN(gen, disc, image):
    ONE = lpm.matrix.Matrix(1, 1)
    ONE[0, 0] = 1

    ZERO = lpm.matrix.Matrix(1, 1)
    ZERO[0, 0] = 0

    # Generate the generator input noise
    noise = lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)], rows=5)

    # Get the discriminator output from the generator
    generated = gen.feedForward(noise)

    # Train the discriminator on a dataset image -- We want the discriminator to output 1 (i.e. is a dataset image)
    disc.backpropagate(image, ONE)
    # wandb.log({"discriminator-loss-DATASET": disc.metrics("loss")[-1]})

    # Train the discriminator on a generated image -- We want the discriminator to output 0 (i.e. is a generated image)
    disc.backpropagate(generated, ZERO)
    # wandb.log({"discriminator-loss-GENERATED": disc.metrics("loss")[-1]})

    # Backpropagate over both networks to train the generator without training the discriminator.
    # The output of this should be 1 as the generator's output through the discriminator should
    # mimic a real dog image
    # network.Network.backpropagateDual(gen, disc, noise, ONE)
    network.Network.backpropagateNonDual(gen, noise)


test = True

data = []

for i in range(1000):
    points = generateNoise(100, std=0 if i != 0 else 0.5)
    x = []
    y = []

    for p in points:
        x.append(p[0])
        y.append(p[1])

    data.append(x + y)

plt.scatter(data[0][:100], data[0][100:])
plt.show()

# The generator network
generator = network.Network(layers=(5, 70, 200), lr=0.02, activations=[lpm.network.TANH] * 2)
discriminator = network.Network(layers=(200, 100, 25, 1), lr=0.02, activations=[lpm.network.TANH] * 3)

discriminator.log("loss", 1)
generator.log("loss", 1)

if not test:
    plotStuff = True

    for i in lpm.progress.Progress(range(1000001)):
        index = random.randint(0, len(data) - 1)
        trainGAN(generator, discriminator, data[index])

        # wandb.log({"generator-loss": generator.metrics("loss")[-1]})

        if i % 50000 == 0 and plotStuff:
            plt.scatter(data[index][:100], data[index][100:])

            noise = lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)], rows=5)
            out = generator.feedForward(noise)

            print("Generator loss:", generator.metrics("loss")[-1])
            print("Discriminator loss:", discriminator.metrics("loss")[-1])
            print("Discriminator output on generated:", discriminator.feedForward(out))
            print("Discriminator output on dataset:", discriminator.feedForward(data[5]))

            plt.scatter(out.toList()[:100], out.toList()[100:])

            plt.show()

    generator.plotMetric("loss")
    discriminator.plotMetric("loss")
else:
    noise = lpm.matrix.Matrix(data=[random.uniform(-1, 1) for _ in range(5)], rows=5)
    preTrain = generator.feedForward(noise)

    for i in lpm.progress.Progress(range(10000)):
        index = random.randint(0, len(data) - 1)
        trainGAN(generator, discriminator, data[index])

    postTrain = generator.feedForward(noise)

    for i in lpm.progress.Progress(range(10000)):
        network.Network.backpropagateDualNoChange(generator, discriminator, noise, lpm.matrix.Matrix(data=[[1]], rows=1))

    finalTrain = generator.feedForward(noise)

    plt.scatter(data[0][:100], data[0][100:])
    plt.scatter(data[1][:100], data[1][100:])
    plt.scatter(preTrain.toList()[:100], preTrain.toList()[100:])
    plt.scatter(postTrain.toList()[:100], postTrain.toList()[100:])
    plt.scatter(finalTrain.toList()[:100], finalTrain.toList()[100:])
    plt.show()

    print("Discriminator on pre-trained     :", discriminator.feedForward(preTrain))
    print("Discriminator on trained         :", discriminator.feedForward(postTrain))
    print("Discriminator on dataset (noisy) :", discriminator.feedForward(data[0]))
    print("Discriminator on dataset (clean) :", discriminator.feedForward(data[1]))

    print("\n\n\n\n\n\n\n")

    network.Network.backpropagateDualNoChange(generator, discriminator, noise, lpm.matrix.Matrix(data=[[1]], rows=1))
    print("\n")
    network.Network.backpropagateNonDualNoChange(generator, noise)
