using Revise
using UTCGP

dir = @__DIR__
pwd_dir = pwd()

# Imports
include(pwd_dir * "/utils/imports.jl")

# Settings
include(pwd_dir * "/utils/utils.jl")

parsed_args = args_parse()
seed_ = parsed_args["seed"]
Random.seed!(seed_)
println("The random seed is : $(seed_)")

# Initialize the project
disable_logging(Logging.Error)
# Python 
python_path = get_python_path() # and builds pycall
dataset_path = get_psb2_path()
n_runs = get_nruns()
Pkg.build("PyCall")

# HASH 
hash = get_unique_id()

# PARAMS --- --- 
N_TRAIN = 200
N_TEST = 2000
PROBLEM = "substitution-cipher"
EXPERIMENT_NAME = "Sub. Cipher"
STAGE = 1
MODE = "MT"
N_types = 3
endpoint = EndpointBatchLevensthein

# LOAD THE DATA --- --- 
load_psb2_data(dataset_path, PROBLEM, N_TRAIN, N_TEST) # loads to python memory

extra_inputs = [""]
# TRAIN DATA --- ---- 
X, Y = make_rows(
    ["input1", "input2", "input3"],
    [string_caster, string_caster, string_caster],
    ["output1"],
    [string_caster],
    extra_inputs,
    "train_data"
)

# TEST DATA --- ---- 
X_test, Y_test = make_rows(
    ["input1", "input2", "input3"],
    [string_caster, string_caster, string_caster],
    ["output1"],
    [string_caster],
    extra_inputs,
    "test_data"
)

@assert unique(length.(X)) == unique(length.(X_test))
@assert length(X) == N_TRAIN
@assert length(X_test) == N_TEST
offset_by = length(X_test[1])

### RUN CONF ###
run_conf = runConf(
    10, n_runs, 1.1, 0.2
)

# Bundles 
string_bundles = get_string_bundles()
liststring_bundles = get_liststring_bundles()
liststringtuples_bundles = get_list_string_tuples_bundles()

# Libraries
lib_string = Library(string_bundles)
lib_liststring = Library(liststring_bundles)
lib_liststringtuples = Library(liststringtuples_bundles)

# MetaLibrary
ml = MetaLibrary([lib_string, lib_liststring, lib_liststringtuples])

### Model Architecture ###
model_arch = modelArchitecture(
    [String, String, String, String],
    [1, 1, 1, 1],
    [String, Vector{String}, Vector{Tuple{String,String}}],
    [String],
    [1]
)


### Node Config ###
N_nodes = 30
println("N Nodes : $N_nodes")
node_config = nodeConfig(N_nodes, 1, 3, offset_by)

### Make UT GENOME ###
shared_inputs, ut_genome = make_evolvable_utgenome(
    model_arch, ml, node_config
)
initialize_genome!(ut_genome)
correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
fix_all_output_nodes!(ut_genome)

# TRACKING
h_params = Dict("connection_temperature" => node_config.connection_temperature,
    "n_nodes" => node_config.n_nodes,
    "lambda" => run_conf.lambda_,
    "budget" => run_conf.generations,
    "mutation_rate" => run_conf.mutation_rate,
    "output_mutation_rate" => run_conf.output_mutation_rate,
    "mode" => MODE,
    "Correction" => "true",
    "output_node" => "fixed",
    "experiment" => EXPERIMENT_NAME,
    "stage" => STAGE,
    "n_train" => N_TRAIN,
    "n_test" => N_TEST,
    "problem" => PROBLEM,
    "seed" => seed_,
    "mutation" => "default_numbered_new_material_mutation_callback"
)
f = open(dir * "/metrics/ut/" * string(hash) * ".json", "a", lock=true)
metric_tracker = jsonTracker(h_params, f)
repeat_metric_tracker = repeatJsonTracker(metric_tracker) # aim_loss_epoch_callback = AIM_LossEpoch(run_jl)
test_tracker = jsonTestTracker(metric_tracker, endpoint, X_test, Y_test)

#######
# FIT #
#######

best_genome, best_program, _ = fit(
    # @enter fit(
    X,
    Y,
    shared_inputs,
    ut_genome,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    [:default_population_callback],
    [:default_numbered_new_material_mutation_callback],
    [:default_ouptut_mutation_callback],
    [:default_decoding_callback],
    # Endpoints
    endpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    [:default_elite_selection_callback],
    # Epoch Callback
    [metric_tracker, test_tracker],
    # Final callbacks ?
    (:default_early_stop_callback,), # 
    repeat_metric_tracker # .. 
)

save_json_tracker(metric_tracker)
close(metric_tracker.file)