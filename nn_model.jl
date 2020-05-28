using Flux
using Flux: @epochs
using Flux.Optimise: ADAM, train!
using Flux.Data: DataLoader
using TextAnalysis
using Random
using CSV

@info "Loading data..."
df = CSV.read("data/imdb_dataset.csv")
# df = df[1:1000, :]

function preprocess(doc::String)
    doc = StringDocument(doc)
    remove_case!(doc)
    prepare!(doc, strip_html_tags | strip_punctuation | strip_non_letters | strip_stopwords)
    stem!(doc)
    return doc
end


@info "Data preprocessing..."
as_binary(x) = Int(x == "positive")
df.sentiment = map(as_binary, df.sentiment);

doc_list = map(preprocess, df.review);
corpus = Corpus(doc_list);

update_lexicon!(corpus);
m = DocumentTermMatrix(corpus);

tfdf = Matrix(tf_idf(m));

samples = size(tfdf, 1)
perm = randperm(samples);
train_perc = 0.75
train_count = Int(floor(train_perc*samples))

train_x = tfdf[perm[1:train_count], :]
test_x = tfdf[perm[train_count:end], :]
train_y = df.sentiment[perm[1:train_count]]
test_y = df.sentiment[perm[train_count:end]]

# data = DataLoader(transpose(tf), hcat(df.sentiment...), shuffle=true, batchsize=64)
train_data = DataLoader(transpose(train_x), hcat(train_y...), shuffle=true, batchsize=64)
test_data = DataLoader(transpose(test_x), hcat(test_y...), shuffle=true, batchsize=64)

input_size = size(tfdf, 2)
epochs = 50

model = Chain(
    Dense(input_size, 100, relu),
    Dropout(0.5),
    Dense(100, 100, relu),
    Dropout(0.5),
    Dense(100, 1),
)

function loss_all(data, model)
    l = 0f0
    for (x, y) in data
        l += Flux.mse(model(x), y)
    end
    l / length(data)
end

@info "Training model..."
loss(x,y) = Flux.mse(model(x), y)
opt = ADAM()

train_loss = []
test_loss = []
for e in 1:epochs
    @info "EPOCH: $e"
    train!(loss, Flux.params(model), train_data, opt)
    push!(train_loss, loss_all(train_data, model))
    push!(test_loss, loss_all(test_data, model))
end

println(train_loss)
println(test_loss)
# println("Train Loss: ")
# for i in train_loss
#     println(i)
# end
# println("Test Loss: ")
# for i in test_loss
#     println(i)
# end
