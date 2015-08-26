using Docile, Docile.Interface, Lexicon, Strada

const api_directory = "api"
const modules = [Strada]

# Stuff from Lexicon.jl:
writeobj(any) = string(any)
writeobj(m::Method) = first(split(string(m), "("))

function url(m)
    line, file = m
    return Base.fileurl(file)
end

function mysave(file::String, m::Module, order = [])
    mysave(file, documentation(m), order)
end

function mysave(file::String, docs::Metadata, order = [:category, :name])
    isfile(file) || mkpath(dirname(file))
    open(file, "w") do io
        info("writing documentation to $(file)")
        println(io)
        for (k,v) in EachEntry(docs, order = order)
            name = writeobj(k)
            source = v.data[:source]
            catgory = category(v)
            comment = catgory == :comment
            println(io)
            println(io)
            !comment && println(io, "## $name")
            println(io)
            println(io, v.docs.data)
            path = last(split(source[2], r"v[\d\.]+(/|\\)"))
            !comment && println(io, "[$(path):$(source[1])]($(url(source)))")
            println(io)
        end
    end
end

cd(dirname(@__FILE__)) do

    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    for mod in modules
        update!(index, save(joinpath(api_directory, "$(mod).md"), mod))
    end
    save(joinpath(api_directory, "index.md"), index; md_subheader = :category)

    # Add a reminder not to edit the generated files.
    open(joinpath(api_directory, "README.md"), "w") do f
        print(f, """
        Files in this directory are generated using the `build.jl` script. Make
        all changes to the originating docstrings/files rather than these ones.

        Documentation should *only* be built directly on the `master` branch.
        Source links would otherwise become unavailable should a branch be
        deleted from the `origin`. This means potential pull request authors
        *should not* run the build script when filing a PR.
        """)
    end

    mysave(joinpath(api_directory, "layers.md"), filter(metadata(Strada); files=["layers.jl"]))

    # info("Adding all documentation changes in $(api_directory) to this commit.")
    # success(`git add $(api_directory)`) || exit(1)

end
