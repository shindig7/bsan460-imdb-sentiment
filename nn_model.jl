using Flux
using Flux: @epochs
using Flux.Optimise: ADAM, train!
using Flux.Data: DataLoader
using TextAnalysis
using CSV

df = CSV.read("data/imdb_dataset.csv")
df = df[1:1000, :]

function preprocess(doc::String)
    doc = StringDocument(doc)
    remove_case!(doc)
    prepare!(doc, strip_html_tags | strip_punctuation | strip_non_letters | strip_stopwords)
    stem!(doc)
    return doc
end


as_binary(x) = Int(x == "positive")
df.sentiment = map(as_binary, df.sentiment);

doc_list = map(preprocess, df.review);
corpus = Corpus(doc_list);

update_lexicon!(corpus);
m = DocumentTermMatrix(corpus);

tf = Matrix(tf_idf(m));
data = DataLoader(transpose(tf), hcat(df.sentiment...), shuffle=true, batchsize=64)

input_size = size(tf, 2)
epochs = 10

model = Chain(
    Dense(input_size, 10, relu),
    Dense(10, 1, sigmoid),
)

function loss_all(data, model)
    l = 0f0
    for (x, y) in data
        l += Flux.binarycrossentropy(model(x), y)
    end
    l / length(data)
end

loss(x,y) = Flux.binarycrossentropy(model(x), y)
opt = ADAM()
evalcb = () -> @show(loss_all(data, model))

@epochs epochs train!(loss, Flux.params(model), data, opt, cb=evalcb)
