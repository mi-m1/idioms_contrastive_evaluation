results = {}


model_name = "gpt"
setting = "figurative"
f1 = 0.7654321

if model_name not in results:
    results[model_name] = {"figurative": [], "literal": []}

results[model_name][setting].append(f1)

print(results)

# figurative mistral7binstructv0.3 p3
# literal mistral7binstructv0.3 p3
# literal mistral7binstructv0.3 p1
# figurative mistral7binstructv0.3 p1
{'flant5xxl_p3': {'figurative': [0.8886498117267349], 'literal': [0.8560354374307863]}, 'flant5xl_p1': {'figurative': [0.996600291403594], 'literal': [0.030505243088655862]}, 'flant5xl_p3': {'figurative': [0.9576185671039354], 'literal': [0.49526584122359796]}, 'llama38binstruct_p3': {'figurative': [0.7889800703399765], 'literal': [0.7650926479378363]}, 'flant5xxl_p2': {'figurative': [0.8227920227920228], 'literal': [0.9512690355329949]}, 'gpt4_p1': {'figurative': [0.9388803287108372], 'literal': [0.9523326572008114]}, 'flant5small_p1': {'figurative': [0.011549566891241578], 'literal': [1.0]}, 'flant5xxl_p1': {'figurative': [0.9372427983539094], 'literal': [0.707942464040025]}, 'gpt35turbo_p2': {'figurative': [0.9311950336264874], 'literal': [0.712772585669782]}, 'llama27bchathf_p2': {'figurative': [0.7135740971357409], 'literal': [0.6470203012442698]}, 'mistral7binstructv0.3_p2': {'figurative': [0.995136186770428], 'literal': [0.19089316987740806]}, 'gpt35turbo_p1': {'figurative': [0.9062996294335627], 'literal': [0.8957776590058792]}, 'gpt35turbo_p3': {'figurative': [0.9777337951509154], 'literal': [0.6143527833668678]}, 'flant5large_p1': {'figurative': [0.9862610402355251], 'literal': [0.09057301293900184]}, 'gpt4o_p1': {'figurative': [0.5712309820193637], 'literal': [0.026743075453677174]}, 'mistral7binstructv0.3_p3': {'figurative': [], 'literal': []}, 'flant5large_p2': {'figurative': [1.0], 'literal': [0.0]}, 'flant5large_p3': {'figurative': [0.9985458070770722], 'literal': [0.01536983669548511]}, 'llama38binstruct_p1': {'figurative': [0.0], 'literal': [1.0]}, 'llama38binstruct_p2': {'figurative': [0.8248009101251422], 'literal': [0.7038895859473023]}, 'flant5small_p2': {'figurative': [0.0], 'literal': [1.0]}, 'gpt4_p3': {'figurative': [0.9443025038323966], 'literal': [0.9250780437044746]}, 'gpt4o_p3': {'figurative': [0.6807151979565773], 'literal': [0.750906892382104]}, 'gpt4_p2': {'figurative': [0.9377892030848329], 'literal': [0.9120589784096893]}, 'llama27bchathf_p3': {'figurative': [0.38341158059467917], 'literal': [0.79533527696793]}, 'flant5xl_p2': {'figurative': [0.9762140733399405], 'literal': [0.5388967468175389]}, 'mistral7binstructv0.3_p1': {'figurative': [], 'literal': []}, 'llama27bchathf_p1': {'figurative': [0.999031007751938], 'literal': [0.0019342359767891683]}, 'flant5small_p3': {'figurative': [0.003864734299516908], 'literal': [1.0]}, 'gpt4o_p2': {'figurative': [0.889247311827957], 'literal': [0.4383975812547241]}}


{'gpt35turbo_': [[0.9062996294335627], [0.9311950336264874], [0.9777337951509154]], 'gpt4_': [[0.9388803287108372], [0.9377892030848329], [0.9443025038323966]], 'gpt4o_': [[0.5712309820193637], [0.889247311827957], [0.6807151979565773]], 'llama27bchathf_': [[0.999031007751938], [0.7135740971357409], [0.38341158059467917]], 'flant5small_': [[0.011549566891241578], [0.0], [0.003864734299516908]], 'flant5large_': [[0.9862610402355251], [1.0], [0.9985458070770722]], 'flant5xl_': [[0.996600291403594], [0.9762140733399405], [0.9576185671039354]], 'flant5xxl_': [[0.9372427983539094], [0.8227920227920228], [0.8886498117267349]], 'mistral7binstructv0.3_': [[], [0.995136186770428], []]}