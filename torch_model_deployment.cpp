#include <torch/script.h>

#include <iostream>
#include <memory>
#include <cstdlib>

using namespace std;

const char* model_pth_env = getenv("SERIALIZED_MODEL_FILE");
string model_path = (model_pth_env != nullptr) ? model_pth_env : "traced_resnet_model.pt";

int main(int argc, const char* argv[]) {

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        cerr << "error loading the model\n";
        return -1;
    }

    cout << "torch module successfuly loaded"<<endl;

    // vector with inputs
    vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its outputs into a tensor
    at::Tensor output = module.forward(inputs).toTensor();
    cout << output.slice(/*dim*/1, /*start*/0, /*end*/5) <<endl;
    
}