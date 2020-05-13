using Flux
using Flux.Optimise: ADAM, train!
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
df.sentiment = map(as_binary, df.sentiment)

doc_list = map(preprocess, df.review)
corpus = Corpus(doc_list)

update_lexicon!(corpus)
m = DocumentTermMatrix(corpus)

tf = Matrix(tf_idf(m))

train_data = zip(tf, df.sentiment)

input_size = size(tf, 2)
epochs = 10

model = Chain(
    Dense(input_size, 10, relu),
    Dense(10, 1),
)


loss(x,y) = Flux.mse(model(x), y)
opt = ADAM()

train!(loss, Flux.params(model), train_data, opt)

println(model(X))
