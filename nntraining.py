from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from tensorflow import keras
from gamecode import Game
import collections.abc
from operator import itemgetter
import pickle

###example at bottom, this is all setup

#this creates a blank keras neural network, play around with the layers and parameters if you want while initializing
#if you change stuff around, you will have to change a variable in the unflatten function to reflect that (it is earmarked)
def modelinit():
    model = Sequential()

    model.add(Dense(16,input_shape = (16,), activation= 'relu'))
    model.add(Dense(60, activation= 'relu'))
    model.add(Dense(40, activation= 'relu'))
    model.add(Dense(30, activation= 'relu'))
    model.add(Dense(4, activation= 'softmax'))
    model.compile(optimizer='adam',loss="categorical_crossentropy", metrics = ["accuracy"])

    return(model)

#this is how a model plays 2048. if watch is set to true, the gameboard will be printed after every move so you can see what the
#model's strategy is. this function returns the final score of the game, which is used as a fitness score for the 
#genetic algorithm
def play_game(nn_model, watch=False):
    a = Game()
    def move(movenum):
        if movenum == 0:
            a.up()
        elif movenum == 1:
            a.left()
        elif movenum == 2:
            a.down()
        elif movenum == 3:
            a.right()
    while a.gameover == False:
        boardlist = []
        for i in range(4):
            for x in range(4):
                boardlist.append(a.board[i][x]) 
        boardlist = np.expand_dims(boardlist, axis=0)
        boardlist = tf.convert_to_tensor(boardlist)
        predictionlist = nn_model.predict(boardlist)
        movecheck = 0 
        whichtry = 0.0
        oldboard = a.board.copy()
        while movecheck == 0:
            inttry = int(whichtry)
            whichmove = np.argsort(predictionlist)[0][3-inttry]
            move(whichmove)
            if watch == True:
                print(a.board)
                print('')
            if np.array_equal(oldboard, a.board) and 3-inttry >= 0:
                whichtry = whichtry +.5
            elif 3-inttry < 0:
                a.random_new()
            else:
                movecheck = 1
                whichtry = 0 
            if a.gameover == True:
                movecheck = 1    
    
    if watch == True:
        print(a.score)
             
    return a.score


#given the weights of a model, this function uses recursion to flatten them into a 1D list. 
# This allows for more granular manipulation
def flatten(weights):
    w = []
    for l in weights:
        if isinstance(l, collections.abc.Iterable):
            w = w + flatten(l)
        else:
            w = w + [l]
    return w

#given a flattened list, this puts the weights back so tht they can be formatted onto a model
def unflatten(weights,model) :
    w = []
    i = 0
    shape = [16,60,40,30,4]##################################if you change the neural network shape, change here to reflect
    for l, size in enumerate(shape):
        layer = model.layers[l].get_weights()
        params = layer[0]
        bias= layer[1]
        new_params = []
        new_bias = []
        for param in params:
            new_params.append(weights[i:i+size])
            i = i+size
        for b in bias:
            new_bias.append(weights[i])
            i += 1
        w.append(np.array(new_params))
        w.append(np.array(new_bias))
    return w

#Not the most elegant or efficient code, but this takes two flattened models, creates another flattened model that is a 
#mix of the two, and then mutates the flattened model. Note that you can pass the same model twice if you
#just want to mutate a model
def crossandmutate(flattened1, flattened2,heavy=False):
    flattenednew = []
    runlength = 0
    runison = False
    for i in range(len(flattened1)):
        chromo = np.random.normal(.5,.13)
        flattenednew.append(chromo*flattened1[i]+(1-chromo)*flattened2[i])
        mut = np.random.random()
        if heavy == False:
            if mut > .8:
                change = -.1 + .2*np.random.random()
                flattenednew[i] += change
            elif mut > .95:
                change = -.4 + .8*np.random.random()
                flattenednew[i] += change
            elif mut > .995:
                change = -.6 + 1.2*np.random.random()
                flattenednew[i] += change
            elif mut > .9996:
                change = -1.2 + 2.4*np.random.random()
                flattenednew[i] += change
        elif heavy == True:
            if runlength == 0:
                isrun = np.random.random()
                if isrun > .999:
                    runison = True
                    if mut > .5:
                        posorneg = 1
                    else:
                        posorneg = -1
            if runison == True:
                change = posorneg*4*np.random.random()
                flattenednew[i] += change
                runlength += 1
                isover = np.random.random()
                chanceover = 1-(runlength*.001)
                if isover > chanceover:
                    runison = False
                    runlength = 0
            if mut > .6:
                change = -.4 + .8*np.random.random()
                flattenednew[i] += change
            elif mut > .9:
                change = -1.2 + 2.4*np.random.random()
                flattenednew[i] += change
            elif mut > .98:
                change = -3 + 6*np.random.random()
                flattenednew[i] += change
            elif mut > .995:
                change = -5 + 10*np.random.random()
                flattenednew[i] += change
    return flattenednew




#This will create a randomized family of models that can be used as a starting point for the algorithm
def startfamily(famsize):
    models = []
    flattenedmodels = []
    for i in range(famsize):
        model = modelinit()
        x = flatten(model.get_weights())
        for a in range(5):
            x = crossandmutate(x,x,True)
        flattenedmodels.append(x)
        newmodelweights = unflatten(x,model)
        model.set_weights(newmodelweights)
        models.append(model)

    pickle.dump(flattenedmodels, open('flattenedgenerations/origmodels.p', 'wb'))
    return(models)



#This is where the genetic algorithm is. 
def evolve(models,numgeneration,savemodelsunderoptional='default',Start = 0, heavymut = False, percentadvance = .1):
    size = len(models)
    gencounter = Start
    numgeneration = gencounter +numgeneration
    while gencounter < numgeneration:
        generation = []
        for i in models:
            mas = [i]
            score = play_game(i)
            mas.append(score)    
            generation.append(mas)


        sortedlist = sorted(generation, key=itemgetter(1), reverse=True)
        flatlist = []
        howmany = int(len(models)*percentadvance)
        for i in range(howmany): 
            flatlist.append(flatten(sortedlist[i][0].get_weights()))

        cutoff = int(howmany*.5)
    
        
        while len(flatlist) < size: 
            reprod = [0,1]
            for x in reprod:
                if np.random.random() > .7:
                    reprod[x] = np.random.randint(0,cutoff)
                else:
                    reprod[x] = np.random.randint(cutoff,howmany)

            p1 = flatlist[reprod[0]]
            p2 = flatlist[reprod[1]]
    
            flatlist.append(crossandmutate(p1,p2,heavymut))

        winner = sortedlist[0][0]
        winnerpath = f'generationwinners/winner{gencounter}.h5'
        genpath = f'flattenedgenerations/{savemodelsunderoptional}.p'
        winner.save(winnerpath)
        pickle.dump(flatlist, open(genpath, 'wb'))

        for x in range(len(flatlist)):
            p = unflatten(flatlist[x],models[x])
            models[x].set_weights(p)
        
        gencounter += 1
        
        print(f'{gencounter}, {sortedlist[0][1]}, {sortedlist[len(sortedlist)-1][1]}')
        
    return(models)
        

#this allows you to load a past generation that you have saved. You can also create a larger or 
#smaller population based on the generation you are loading
def load_generation(flatfilename, newsize = 0, ):
    location = f'flattenedgenerations/{flatfilename}.p'
    flatgen = pickle.load(open(location,'rb'))
    loaded = []
    blankmodel = modelinit()

    if newsize < len(flatgen):
        flatgen = flatgen[:newsize]
    
    while len(flatgen) < newsize:
        reprod = [0,1]
        for x in reprod:
            reprod[x] = np.random.randint(0,len(flatgen))

        p1 = flatgen[reprod[0]]
        p2 = flatgen[reprod[1]]
        flatgen.append(crossandmutate(p1,p2,heavy=True))

    for i in flatgen:
        newmod = unflatten(i,blankmodel)
        blankmodel.set_weights(newmod)
        loaded.append(blankmodel)
    return(loaded)

#This allows you to load a generation from a single model
def load_from_model(modelpath, gensize):
    light = int(gensize/2)
    heavy = gensize -light
    models = []
    for a in range(light):
        model = tf.keras.models.load_model(modelpath)
        fmodel = flatten(model.get_weights())
        for b in range(3):
            fmodel = crossandmutate(fmodel,fmodel,False)
        model = model.set_weights(unflatten(fmodel,model))
        models.append(model)
    for a in range(heavy):
        model = tf.keras.models.load_model(modelpath)
        fmodel = flatten(model.get_weights())
        for b in range(6):
            fmodel = crossandmutate(fmodel,fmodel,True)
        model = model.set_weights(unflatten(fmodel,model))
        models.append(model)
    return(models)





###EXAMPLE###
'''
~~~here we initialize a population of 50 random neural models~~~

my_models = startfamily(50)

~~~here we evolve the models through 60 generations. we save the latest generation under 'mymodels', ~~~
~~~we are heavily mutating each new model, and each round the 7 top models are andvancing and reproducing ~~~

my_models = evolve(my_models,60,'mymodels',0,True,.14)

~~~we load the best model of the last generation we ran above, then we can watch it play the game~~~

best_model = tf.keras.models.load_model(r"generationwinners\winner59.h5")
play_game(best_model,watch = True)
'''