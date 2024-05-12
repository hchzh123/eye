import torch.onnx
from train import SELFMODEL
#Function to Convert to ONNX
# 模型查看工具:https://netron.app/
def Convert_ONNX(model, inputsize=224):

    # set the model to inference mode
    model.eval()
    # Let's create a dummy input tensor
    # dummy_input = torch.randn(1, input_size, requires_grad=True)
    dummy_input = torch.randn(1, 3, inputsize, inputsize, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ImageClassifier.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    # todo 加载模型
    model_name = "resnet50d"
    num_classes = 5
    model_path = "../../checkpoints/resnet50d_pretrained_224/resnet50d_10epochs_accuracy0.99501_weights.pth"
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()

    # model.to(device)

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX(model=model)