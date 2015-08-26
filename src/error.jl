function error_callback(msg::Ptr{Uint8})
	message = bytestring(msg)
	error("Caffe signaled an error: " * message)
end

const error_callback_c = cfunction(error_callback, Void, (Ptr{Uint8},))
