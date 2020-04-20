# bot.py
import os
import random

import discord
from discord.ext import commands
from dotenv import load_dotenv
import numpy
import tensorflow
import tflearn
import nltk
import random
import json
import pickle
import asyncio
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.load("model.tflearn")
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, bagwords):
    bag = [0 for _ in range(len(bagwords))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(bagwords):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(inp):
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    print(random.choice(responses))
    # return random.choice(responses)


# -------------------
io = False
lx = False
ls = 0
t = ""
pt = ""
r = ""

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

bot = commands.Bot(command_prefix='timmy ')


@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
    await bot.change_presence(status=discord.Status.do_not_disturb, activity=discord.Activity(type=discord.ActivityType.watching, name='for "timmy help"'))


@bot.command(name='chat')
@commands.has_role('Admin')
async def chat(ctx, on=True):
    global io, lx
    if on:
        io = True
        if lx:
            lx = False
            await ctx.send('`Learning Disabled`')
        await ctx.send('`Chat Enabled`')
    else:
        io = False
        await ctx.send('`Chat Disabled`')


@bot.command(name='learn')
@commands.has_role('Admin')
async def learn(ctx, on=True):
    global lx, io
    if on:
        lx = True
        if io:
            io = False
            await ctx.send('`Chat Disabled`')
        await ctx.send('`Learning Enabled`')
        await ctx.send('Start off by giving me a category. '
                       'Say `timmy categories` to see the present ones, or type a new one.')
    else:
        lx = False
        await ctx.send('`Learning Disabled`')


@bot.command(name='categories')
async def categories(ctx):
    tags = "Categories: "
    with open('intents.json') as json_file:
        dataset = json.load(json_file)
        for p in dataset['intents']:
            tags += "`" + p['tag'] + "`, "
#            print('Name: ' + p['tag'])
#            print('Website: ', *p['patterns'], sep=", ")
#            print('From: ', *p['responses'])
#            print('')
    await ctx.send(tags[:-2])


@bot.command(name='test')
async def test(ctx):
    with open('intents.json', 'rt+') as f:
        print(f.read())


@bot.command(name='remodel')
@commands.has_role('Admin')
async def remodel(ctx):
    global data, model
    await ctx.send('Remodeling...')

    with open("intents.json") as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

    model.load("model.tflearn")
    await ctx.send('Complete')


@bot.event
async def on_message(message):
    global ls, t, pt, r
    if message.author == bot.user:
        return
    if message.channel != bot.get_channel(700127574935863310):
        return
    if message.content.startswith('timmy'):
        await bot.process_commands(message)
        return
    async with message.channel.typing():
        await asyncio.sleep(1)
    if io:
        # response = chat(message.content)
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)

        results = model.predict([bag_of_words(message.content, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        response = random.choice(responses)
        await message.channel.send(response)
    elif lx:
        await bot.process_commands(message)
        if lx and ls == 0:
            t = message.content
            ls = 1
            await message.channel.send('Now say something to me.')
            return
        elif lx and ls == 1:
            pt = message.content
            ls = 2
            await message.channel.send('How should I respond?')
            return
        elif lx and ls == 2:
            r = message.content
            with open('intents.json', 'rt+') as f:
                d = json.load(f)
                included = False
                for p in d['intents']:
                    if p["tag"] == t:
                        p["patterns"].append(pt)
                        p["responses"].append(r)
                        included = True
                if not included:
                    d["intents"].append({"tag": t, "patterns": [pt], "responses": [r]})
                f.seek(0)
                f.truncate(0)
                json.dump(d, f, indent=2)
            await message.channel.send('I learned to respond to ' + pt + ' with ' + r + '.'
                                       ' You must use the command `timmy remodel` for the changes to take effect')
            ls = 0
            return
    else:
        return


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send('`You do not have permission for this command.`')

bot.run(TOKEN)
