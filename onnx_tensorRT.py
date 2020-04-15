import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnx

def save_onnx(model,onnx_path,input_names,output_names):
    dummy_input = torch.randn(1, 3, 640, 640, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, verbose=False,
    input_names=input_names,output_names=output_names)

def onnx_build_engine(onnx_file_path):
    ##load serial file
    # if os.path.exists(onnx_file_path[:-4]+'rt'):
    #     with open(onnx_file_path[:-4]+'rt', 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    #     return engine

    # print log
    G_LOGGER = trt.Logger(trt.Logger.WARNING)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(G_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 20

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        # # save serial file
        # with open(onnx_file_path[:-4]+'rt', "wb") as f:
        #     f.write(engine.serialize())

        return engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    context = engine.create_execution_context()
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream, context

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == '__main__':

    ##pytorch-model transform onnx-model
    # input_names = ['input']
    # output_names = ['boxes','classes','landmark']
    # save_onnx(model,'example.onnx',input_names,output_names)

    # load image data
    img_raw = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw,(640,640))
    img = np.float32(img_raw)
    img = img.transpose(2, 0, 1)

    # load engine
    engine = onnx_build_engine('your-model.onnx')
    inputs, outputs, bindings, stream, context = allocate_buffers(engine)

    # do inference
    inputs[0].host = np.ravel(img)
    output = do_inference(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # analytical output
    loc = torch.from_numpy(output[0]).view(1, -1, 4).cuda()
    landms = torch.from_numpy(output[1]).view(1, -1, 10).cuda()
    conf = torch.from_numpy(output[2]).view(1, -1, 2).cuda()
