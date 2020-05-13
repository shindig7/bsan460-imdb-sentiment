# using Flux
# using Flux.Optimise: ADAM, train!
using TextAnalysis
using CSV

df = CSV.read("data/imdb_dataset.csv")

as_binary(x) = Int(x == "positive")
df.sentiment = map(as_binary, df.sentiment)


"""
input_size = 2
epochs = 10

model = Chain(
    Dense(input_size, 10, relu),
    Dense(10, 1),
)

X = [[1, 3], [2,4],[3,6]]
Y = [[5],[6],[7]]

train_data = Flux.Data.DataLoader()
data = zip(hcat(X...), hcat(Y...))

loss(x,y) = Flux.mse(model(x), y)
opt = ADAM()

@epochs 10 train!(loss, Flux.params(model), data, opt)

println(model(X))
"""
