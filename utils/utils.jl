import Pkg
import UUIDs

function get_python_path()
    python_path = ENV["UTCGP_PYTHON"]
    ENV["PYTHON"] = python_path
    return python_path
end

function get_psb2_path()
    dataset_path = ENV["UTCGP_PSB2_DATASET_PATH"]
    return dataset_path
end

function get_nruns()
    n_runs = ENV["UTCGP_NRUNS"]
    n_runs = parse(Int, n_runs)
end

function get_unique_id()
    return UUIDs.uuid4().value
end

function load_psb2_data(dataset_path::String, pb::String, n_train::Int, n_test::Int)
    PROBLEM = pb
    N_TRAIN = n_train
    N_test = n_test

    py"""
    import psb2 
    import numpy as np

    (train_data, test_data) = psb2.fetch_examples(
        $dataset_path, $PROBLEM, $N_TRAIN, $N_TEST, format="psb2", 
    )
    """
end

function make_rows(
    in_keys::Vector{String},
    in_casters::Vector,
    out_keys::Vector{String},
    out_casters::Vector,
    extra_inputs::Vector,
    df_in_python_memory::String)
    X = []
    Y = []
    df = df_in_python_memory
    for x in py"$$df"
        ins = [caster(x[k]) for (caster, k) in zip(in_casters, in_keys)]
        outs = [caster(x[k]) for (caster, k) in zip(out_casters, out_keys)]
        push!(X, Any[ins..., extra_inputs...])
        push!(Y, identity.([outs...]))
    end
    return X, Y
end

function fix_all_output_nodes!(ut_genome::UTGenome)
    for (ith_out_node, output_node) in enumerate(ut_genome.output_nodes)
        to_node = output_node[2].highest_bound + 1 - ith_out_node
        set_node_element_value!(output_node[2],
            to_node)
        set_node_freeze_state(output_node[2])
        println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        println("Output Node material : $(node_to_vector(output_node))")
    end
end

function args_parse()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--seed", "-s"
        help = "Random seed"
        arg_type = Int
        required = true
    end
    parsed_args = parse_args(s)
    return parsed_args
end
