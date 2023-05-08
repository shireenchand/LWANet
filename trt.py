import torch.onnx
import tensorrt as trt
import onnx


def convert(model, imgL, imgR):
	# torch to onnx
	output_path = "model.onnx"
	torch.onnx.export(model, (imgL.cuda(),imgR.cuda()), output_path)

	# onnx to tensorrt
	logger = trt.Logger(trt.Logger.WARNING)
	builder = trt.Builder(logger)
	network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
	parser = trt.OnnxParser(network, logger)
	success = parser.parse_from_file('model.onnx')
	for idx in range(parser.num_errors):
	    print(parser.get_error(idx))

	if not success:
	    print("can't read file")

	config = builder.create_builder_config()
	config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2**30) # 1 MiB
	serialized_engine = builder.build_serialized_network(network, config)
	with open("model.engine", "wb") as f:
	    f.write(serialized_engine)