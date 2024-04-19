using Revise
using UTCGP

using UTCGP: SN_writer, sn_strictphenotype_hasher
import SearchNetworks as sn
import DataStructures: OrderedDict
using UUIDs
import DBInterface

DB_FOLDER = "/dbs/ut/"

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
PROBLEM = "basement"
EXPERIMENT_NAME = "Basement"
STAGE = 1
MODE = "MT"
N_types = 2
endpoint = EndpointBatchAbsDifference

# LOAD THE DATA --- --- 
load_psb2_data(dataset_path, PROBLEM, N_TRAIN, N_TEST) # loads to python memory

extra_inputs = [-1, 0, 1]
# TRAIN DATA --- ---- 
X, Y = make_rows(
    ["input1"],
    [listinteger_caster],
    ["output1"],
    [integer_caster],
    extra_inputs,
    "train_data"
)

# TEST DATA --- ---- 
X_test, Y_test = make_rows(
    ["input1"],
    [listinteger_caster],
    ["output1"],
    [integer_caster],
    extra_inputs,
    "test_data"
)

X_test_behavior = deepcopy(X)
@assert unique(length.(X)) == unique(length.(X_test))
@assert length(X) == N_TRAIN
@assert length(X_test) == N_TEST
offset_by = length(X_test[1])

# SN DB  --- --- 
db_name = string(UUIDs.uuid4().value) * ".db"
db_name = dir * DB_FOLDER * db_name
@show db_name
con = sn.create_DB(db_name)
sn.create_SN_tables!(
    con,
    extra_nodes_cols=OrderedDict(
        "gen_hash" => sn.SN_col_type(string=true),
        "phen_hash" => sn.SN_col_type(string=true),
        "behavior_hash" => sn.SN_col_type(string=true),
        "db_name" => sn.SN_col_type(string=true),
    ),
    extra_edges_cols=OrderedDict(
        "fitness" => sn.SN_col_type(float=true),
        "is_elite" => sn.SN_col_type(float=true),
        "db_name" => sn.SN_col_type(string=true),
        "seed" => sn.SN_col_type(string=true),
    )
)
sn_writer_callback = SN_writer(
    con,
    all_edges(),
    OrderedDict(
        "gen_hash" => sn_genotype_hasher(),
        "phen_hash" => sn_strictphenotype_hasher(),
        "behavior_hash" => sn_behavior_hasher(
            X_test_behavior),
        "db_name" => sn_db_name_node(db_name)
    ),
    OrderedDict(
        "fitness" => sn_fitness_hasher(),
        "is_elite" => sn_elite_hasher(),
        "db_name" => sn_db_name_edge(db_name),
        "seed" => sn_db_name_edge(string(seed_))
    )
)

### RUN CONF ###
run_conf = runConf(
    10, n_runs, 1.1, 0.2
)

# Bundles Integer
integer_bundles = get_integer_bundles()
listinteger_bundles = get_listinteger_bundles()

# Libraries
lib_int = Library(integer_bundles)
lib_list_int = Library(listinteger_bundles)


# MetaLibrary
ml = MetaLibrary([lib_int, lib_list_int])

### Model Architecture ###
model_arch = modelArchitecture(
    [Vector{Int}, Int, Int, Int],
    [2, 1, 1, 1],
    [Int, Vector{Int}],
    [Int],
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
    X,  # multi input
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
    [:default_numbered_new_material_mutation_callback], #[:default_numbered_mutation_callback],
    [:default_ouptut_mutation_callback],
    [:default_decoding_callback], #[:default_free_decoding_callback], #[:default_decoding_callback],
    # Endpoints
    endpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    [:default_elite_selection_callback],
    # Epoch Callback
    [metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    (:default_early_stop_callback,), # 
    repeat_metric_tracker # .. 
)

save_json_tracker(metric_tracker)
close(metric_tracker.file)

sn._execute_command(con, "CHECKPOINT")

nodes = sn._execute_command(con, "select count(*) from nodes")
edges = sn._execute_command(con, "select count(*) from edges")
@show "n nodes $nodes"
@show "n edges $edges"

DBInterface.close!(con)
close(con)