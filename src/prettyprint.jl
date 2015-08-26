import Base.print
import Base.show
import Base.norm
using ProtoBuf
using DataStructures
using Strada
using Compat

print_list(io::IO, elts::Vector, sep::String) = begin
	if length(elts) == 0
		return
	end
	print(io, elts[1])
	for i = 2:length(elts)
		print(io, sep)
		print(io, elts[i])
	end
end

print_dict(io::IO, dict::OrderedDict, equal::ASCIIString="=", sep::ASCIIString=", ") = begin
	maps = ASCIIString[]
	for (key, val) in dict
		push!(maps, string(key) * equal * string(val))
	end
	print_list(io, maps, sep)
end

print(io::IO, layer::Layer) = begin
	param = layer.param
	typ = ProtoBuf.get_field(param, :_type)
	name = ProtoBuf.get_field(param, :name)
	bottoms = ProtoBuf.get_field(param, :bottom)
	print(io, typ, "(:", name, ", [")
	print_list(io, map(string, bottoms), ", ")
	print(io, "]")
	if ProtoBuf.has_field(param, layer.param_name)
		print(io, ", ")
		layer_param = ProtoBuf.get_field(param, layer.param_name)
		print_dict(io, get_active_param_fields(layer_param))
	end
	print(io, ")")
end

print(io::IO, net::CaffeNet) = begin
	for layer in net.layer_defs
		print(io, layer, "\n")
	end
end

print(io::IO, blobs::NetData{CaffeDict}) = begin
	if length(blobs.data) == 0
		return
	end
	layer_name = "layer_name"
	len = max(maximum(map(length, keys(blobs.data))), length(layer_name))
	Base.print(io, rpad(layer_name, len), " : [param_norm grad_norm ] for all blob parameters\n")
	for layer in keys(blobs.data)
		Base.print(io, rpad(layer, len), " : ")
		for i in 1:length(blobs.data[layer])
			thedata = blobs.data[layer][i][:]
			thediff = blobs.diff[layer][i][:]
			theta_fmt = @sprintf "%4.4E" norm(thedata)
			grad_fmt = @sprintf "%4.4E" norm(thediff)
			Base.print(io, "[", theta_fmt, " ", grad_fmt, "] ")
		end
		Base.print(io, "\n")
	end
end

print(io::IO, blob::ApolloDict) = begin
	if length(keys(blob)) == 0
		Base.print(io, "[]\n")
		return
	end
	blob_name = "blob_name"
	len = max(maximum(map(length, keys(blob))), length(blob_name))
	for key in keys(blob)
		Base.print(io, rpad(key, len), " : ")
		theta_fmt = @sprintf "%4.4E" norm(blob[key])
		Base.print(io, "[", theta_fmt, "]\n")
	end
end

norm(x::Array{Float32,0}) = x[1]

print(io::IO, blob::NetData{ApolloDict}) = begin
	if length(keys(blob.data)) == 0
		Base.print(io, "[]\n")
		return
	end
	blob_name = "blob_name"
	len = max(maximum(map(length, keys(blob.data))), length(blob_name))
	for key in keys(blob.data)
		Base.print(io, rpad(key, len), " : ")
		theta_fmt = @sprintf "%4.4E" norm(blob.data[key])
		grad_fmt = @sprintf "%4.4E" norm(blob.diff[key])
		Base.print(io, "[", theta_fmt, " ", grad_fmt, "]\n")
	end
end

print(io::IO, net::ApolloNet) = begin
	Base.print(io, "parameter:\n")
	print(io, net.params)
	Base.print(io, "blobs:\n")
	print(io, net.blobs)
end

show(io::IO, layer::Layer) = print(io, layer)
show(io::IO, blob::ApolloDict) = print(io, blob)
show(io::IO, blob::NetData{ApolloDict}) = print(io, blob)
show(io::IO, net::CaffeNet) = print(io, net)
show(io::IO, net::ApolloNet) = print(io, net)
show(io::IO, blobs::NetData{CaffeDict}) = print(io, blobs)

get_active_param_fields(param) = begin
	dict = OrderedDict{Symbol, Any}()
	for field in fieldnames(param)
		if ProtoBuf.has_field(param, field)
			val = ProtoBuf.get_field(param, field)
			if isa(val, Integer)
				dict[field] = convert(Int, val)
			elseif typeof(val) == Strada.FillerParameter
				dict[field] = string(ProtoBuf.get_field(val, :_type))
			elseif typeof(val) == Strada.BlobShape
				dict[field] = string(ProtoBuf.get_field(val, :dim))
			else
			end
		end
	end
	return dict
end


