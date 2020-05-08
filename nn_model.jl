using Flux
using Flux.Optimise: ADAM, train!

input_size = 2
epochs = 10

model = Chain(
    Dense(input_size, 10, relu),
    Dense(10, 1),
)

X = [[1, 3], [2,4],[3,6]]
Y = [[5],[6],[7]]

data = Flux.Data.DataLoader(hcat(X...), hcat(Y...))

loss(x,y) = Flux.mse(model(x), y)
opt = ADAM()

for e in epochs
    train!(loss, Flux.params(model), data, opt)
end

println(model(X))
