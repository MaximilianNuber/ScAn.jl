module ScAn

using DataFrames, CSV
using Transducers
using BangBang.NoBang: SingletonVector
using BangBang: append!!, union!!
using SplittablesBase: amount

using SparseArrays
using CategoricalArrays
using Base.Threads
using Base

using GLM
using MixedModels

using Statistics
using StatsBase
using Muon

export subset_adata
export normalize_total!
export _lme_ad
export _lme_trunc_ad
export _logit_mm_ad

function subset_adata(data::AnnData, subset_inds::Union{Int, Vector{Int}, Vector{Bool}, UnitRange}, ::Val{:cells})
    #adata.ncells = length(subset_inds)

    adata = deepcopy(data)

    mat = transpose(adata.X)
    mat = mat[:, subset_inds]
    mat = transpose(mat)
    mat = convert(SparseArrays.SparseMatrixCSC, mat)

    adata.X = mat

    adata.obs_names = adata.obs_names[subset_inds]

    if nrow(adata.obs) > 0 #&& nrow(adata.var) > 0
        adata.obs = adata.obs[subset_inds,:]
    end

    if length(adata.layers) > 0
        for key in keys(adata.layers)
            adata.layers[key] = setindex!(adata.layers, adata.layers[key][subset_inds,:], key)
        end
    end

    if length(adata.obsm) > 0
        for key in keys(adata.obsm)
            adata.obsm[key] = adata.obsm[key][subset_inds,:]
        end
    end

    if length(adata.obsp) > 0
        for key in keys(adata.obsp)
            adata.obsp[key] = adata.obsp[key][subset_inds,subset_inds]
        end
    end

    return adata
end

function normalize_total2(mat::AbstractMatrix; target_sum = 1e4, pseudocount = 1)

    chunk_size = length(1:size(mat, 2)) ÷ nthreads()
    chunks = Iterators.partition(1:size(mat, 2), chunk_size)

    submats = map(x -> mat[:, x], chunks);
    tasks = map(submats) do x
        Threads.@spawn begin 
            sum_val = sum(x, dims = 1)
            for j in 1:size(x,2)
                sum_val[j] == 0 && continue
                for i in 1:size(x, 1)
                    x[i, j] = log(x[i, j] / sum_val[j] *10000 +1)
                end
            end
        return x
        end
    end
    res = fetch.(tasks)
    return hcat(res...)
end

function normalize_total!(data::Muon.AnnData; target_sum = 1e4, pseudocount = 1)
    new = normalize_total2(transpose(data.X); target_sum = target_sum, pseudocount = pseudocount)
    data.X = copy(transpose(new))
    return data
end

function get_bin_mat(mat::AbstractMatrix)
    return Int64.(mat .> 0)
end

function _single_lme(exprs::AbstractVector, data::AbstractDataFrame, group, samp, cov_names = nothing)
    exp = Array(exprs)
    if cov_names !== nothing
        model_df = data[:, cov_names]
        model_df.group = data[:, group]
        model_df.samp = data[:, samp]
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))
    else
        model_df = DataFrame(
            group = data[:, group],
            samp = data[:, samp]
        )
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))])
    end

    model_df.expression = exp

    # fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))

    try
        full = MixedModels.fit(LinearMixedModel, fm, model_df)

        arr = DataFrame(coeftable(full))[2, 2:5] |> Array
        arr = append!!(arr, [loglikelihood(full), dof(full)])

        return arr
    catch
        return [NaN, NaN, NaN, NaN, NaN, NaN]
    end
end

function _lme_ad(data::Muon.AnnData, group, samp, cov_names = nothing)
    res = foldxt(
        hcat,
        Map(i -> _single_lme(NK.X[:, i], NK.obs, "classification HF", "dataset", cov_names)),
        1:size(data.X, 2)
    )

    res = permutedims(res)
    res = DataFrame(res, ["Coef.", "Std. Error", "z", "Pr(>|z|)", "logLik", "DoF"])

    res.gene = data.var[:, "Symbol"]

    return res
end

function _single_lme_trunc(exprs::AbstractVector, data::AbstractDataFrame, group, samp, cov_names = nothing)
    exp = Array(exprs)
    if cov_names !== nothing
        model_df = data[:, cov_names]
        model_df.group = data[:, group]
        model_df.samp = data[:, samp]
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))
    else
        model_df = DataFrame(
            group = data[:, group],
            samp = data[:, samp]
        )
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))])
    end

    model_df.expression = exp

    model_df = model_df[collect(model_df.expression .> 0), :]

    # fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))

    try
        full = MixedModels.fit(LinearMixedModel, fm, model_df)

        arr = DataFrame(coeftable(full))[2, 2:5] |> Array
        arr = append!!(arr, [loglikelihood(full), dof(full)])

        return arr
    catch
        return [NaN, NaN, NaN, NaN, NaN, NaN]
    end
end

function _lme_trunc_ad(data::Muon.AnnData, group, samp, cov_names = nothing)
    res = foldxt(
        hcat,
        Map(i -> _single_lme_trunc(NK.X[:, i], NK.obs, "classification HF", "dataset", cov_names)),
        1:size(data.X, 2)
    )

    res = permutedims(res)
    res = DataFrame(res, ["Coef.", "Std. Error", "z", "Pr(>|z|)", "logLik", "DoF"])

    res.gene = data.var[:, "Symbol"]

    return res
end

function _single_logit_mm(exprs::AbstractVector, data::AbstractDataFrame, group, samp, cov_names = nothing; fast = false)
    exp = Array(exprs)
    exp = convert(Vector{Float64}, exp)
    exp = (collect(Map(x -> ifelse(x == 0, 0, 1)), exp))

    if cov_names !== nothing
        model_df = data[:, cov_names]
        model_df.group = data[:, group]
        model_df.samp = data[:, samp]
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))
    else
        model_df = DataFrame(
            group = data[:, group],
            samp = data[:, samp]
        )
        fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))])
    end

    model_df.expression = exp

    # fm = term(Symbol("expression")) ~ sum([term(1), term(Symbol("group")), (term(1) | term(Symbol("samp")))]) + sum(term.(Symbol.(cov_names)))

    try
        full = MixedModels.fit(GeneralizedLinearMixedModel, fm, model_df, Bernoulli(); fast = fast)

        arr = DataFrame(coeftable(full))[2, 2:5] |> Array
        arr = append!!(arr, [loglikelihood(full), dof(full)])

        return arr
    catch
        return [NaN, NaN, NaN, NaN, NaN, NaN]
    end
end

function _logit_mm_ad(data::Muon.AnnData, group, samp, cov_names = nothing; fast = false, base_size = amount(1:size(data.X, 2)) ÷ nthreads())
    res = foldxt(
        hcat,
        Map(i -> _single_logit_mm(NK.X[:, i], NK.obs, "classification HF", "dataset", cov_names; fast = fast)),
        1:size(data.X, 2);
        basesize = base_size
    )

    res = permutedims(res)
    res = DataFrame(res, ["Coef.", "Std. Error", "z", "Pr(>|z|)", "logLik", "DoF"])

    res.gene = data.var[:, "Symbol"]

    return res
end
end

