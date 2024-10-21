#include <torch/script.h>

#include <iostream>
#include <memory>

using namespace std;

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

    cout << "ok"<<endl;
}