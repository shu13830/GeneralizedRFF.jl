using Random, Distributions, LinearAlgebra, SpecialFunctions, HypergeometricFunctions
using Plots, Statistics

# 対称安定分布のサンプリング用の補助関数
function sample_symmetric_stable(α::Float64, d::Int, n::Int)
    """
    α-安定分布からのサンプリング (Definition 1, Proposition 1)
    """
    samples = zeros(n, d)
    
    for i in 1:n
        # 標準ガウシアンベクトル N
        N = randn(d)
        
        # 補助変数 U1, U2, W, Θ
        U1 = rand()
        U2 = rand()
        W = -log(U1)  # 標準指数分布
        Θ = π * (U2 - 0.5)  # [-π/2, π/2] の一様分布
        
        # A_α の計算 (式 14)
        if α ≈ 2.0
            A_α = 1.0  # ガウシアンの場合
        else
            numerator = sin(α*π/4 + α*Θ/2) * (cos(Θ))^(2/α)
            denominator = cos(α*π/4 + (α/2 - 1)*Θ) * W^(2/α - 1)
            A_α = numerator / denominator
        end
        
        # S_α = √(2A_α) * N (式 13)
        samples[i, :] = sqrt(2 * A_α) * N
    end
    
    return samples
end

# 1. 指数べき乗カーネル (Exponential Power Kernel)
function sample_exponential_power_rff(α::Float64, d::Int, M::Int)
    """
    指数べき乗カーネル K(u) = exp(-||u||^α) のRFF
    R = 1 (定数), λ = 1
    """
    println("Exponential Power Kernel (α=$α)")
    
    # η = (λR)^(1/α) * S_α = 1^(1/α) * S_α = S_α
    projections = sample_symmetric_stable(α, d, M)
    
    return projections
end

# 2. 一般化コーシーカーネル (Generalized Cauchy Kernel)
function sample_generalized_cauchy_rff(α::Float64, β::Float64, d::Int, M::Int)
    """
    一般化コーシーカーネル K(u) = 1 / (1 + ||u||^α / (2β))^β
    R ~ Gamma(β), λ = 1/(2β)
    """
    println("Generalized Cauchy Kernel (α=$α, β=$β)")
    
    projections = zeros(M, d)
    
    for i in 1:M
        # R ~ Gamma(β)
        R = rand(Gamma(β, 1.0))
        
        # λR = R/(2β)
        λR = R / (2 * β)
        
        # S_α をサンプル
        S_α = sample_symmetric_stable(α, d, 1)[1, :]
        
        # η = (λR)^(1/α) * S_α
        projections[i, :] = (λR)^(1/α) * S_α
    end
    
    return projections
end

# 3. 一般化マテルンカーネル (Generalized Matérn Kernel)
function sample_generalized_matern_rff(α::Float64, β::Float64, d::Int, M::Int)
    """
    一般化マテルンカーネル
    R ~ InverseGamma(β), λ = β/2
    """
    println("Generalized Matérn Kernel (α=$α, β=$β)")
    
    projections = zeros(M, d)
    
    for i in 1:M
        # R ~ 1/Gamma(β) (逆ガンマ分布)
        G = rand(Gamma(β, 1.0))
        R = 1.0 / G
        
        # λR = (β/2) * R
        λR = (β / 2) * R
        
        # S_α をサンプル
        S_α = sample_symmetric_stable(α, d, 1)[1, :]
        
        # η = (λR)^(1/α) * S_α
        projections[i, :] = (λR)^(1/α) * S_α
    end
    
    return projections
end

# 4. クンマーカーネル (Kummer Kernel)
function sample_kummer_rff(α::Float64, β::Float64, γ::Float64, d::Int, M::Int)
    """
    クンマーカーネル K(u) = M(β, β+γ, -||u||^α)
    R ~ Beta(β, γ), λ = 1
    """
    println("Kummer Kernel (α=$α, β=$β, γ=$γ)")
    
    projections = zeros(M, d)
    
    for i in 1:M
        # R ~ Beta(β, γ)
        R = rand(Beta(β, γ))
        
        # λR = R (λ = 1)
        λR = R
        
        # S_α をサンプル
        S_α = sample_symmetric_stable(α, d, 1)[1, :]
        
        # η = (λR)^(1/α) * S_α
        projections[i, :] = (λR)^(1/α) * S_α
    end
    
    return projections
end

# 5. ベータカーネル (Beta Kernel)
function sample_beta_kernel_rff(α::Float64, β::Float64, γ::Float64, d::Int, M::Int)
    """
    ベータカーネル
    R ~ -log(Beta(β, γ)), λ = 1
    """
    println("Beta Kernel (α=$α, β=$β, γ=$γ)")
    
    projections = zeros(M, d)
    
    for i in 1:M
        # R ~ -log(Beta(β, γ))
        B = rand(Beta(β, γ))
        R = -log(B)
        
        # λR = R (λ = 1)
        λR = R
        
        # S_α をサンプル
        S_α = sample_symmetric_stable(α, d, 1)[1, :]
        
        # η = (λR)^(1/α) * S_α
        projections[i, :] = (λR)^(1/α) * S_α
    end
    
    return projections
end

# 6. トリコミカーネル (Tricomi Kernel)
function sample_tricomi_rff(α::Float64, β::Float64, γ::Float64, d::Int, M::Int)
    """
    トリコミカーネル
    R ~ F(2β, 2γ), λ = 1
    """
    println("Tricomi Kernel (α=$α, β=$β, γ=$γ)")
    
    projections = zeros(M, d)
    
    for i in 1:M
        # R ~ F(2β, 2γ) フィッシャー分布
        # F = (γ * G_β) / (β * G_γ)
        G_β = rand(Gamma(β, 1.0))
        G_γ = rand(Gamma(γ, 1.0))
        R = (γ * G_β) / (β * G_γ)
        
        # λR = R (λ = 1)
        λR = R
        
        # S_α をサンプル
        S_α = sample_symmetric_stable(α, d, 1)[1, :]
        
        # η = (λR)^(1/α) * S_α
        projections[i, :] = (λR)^(1/α) * S_α
    end
    
    return projections
end

# RFFカーネル近似の計算
function compute_rff_kernel(X1, X2, projections)
    """
    RFF近似カーネル値の計算
    K(x1, x2) ≈ (1/M) * Σ cos(η_m^T (x1 - x2))
    """
    M = size(projections, 1)
    n1, n2 = size(X1, 1), size(X2, 1)
    K = zeros(n1, n2)
    
    for i in 1:n1
        for j in 1:n2
            diff = X1[i, :] - X2[j, :]
            cosine_values = [cos(dot(projections[m, :], diff)) for m in 1:M]
            K[i, j] = mean(cosine_values)
        end
    end
    
    return K
end

# テスト実行
# function test_kernels()
#     Random.seed!(42)
    
#     # パラメータ設定
#     d = 2  # 次元数
#     M = 1000  # RFF射影数
    
#     # テストデータ
#     n_test = 50
#     x_range = range(-3, 3, length=n_test)
#     X_test = [[x, y] for x in x_range[1:5:end], y in x_range[1:5:end]]
#     X_test = hcat([[p[1], p[2]] for p in vec(X_test)]...)'
    
#     println("=== カーネルRFF実装テスト ===\n")
    
#     # 1. 指数べき乗カーネル (α=1.5)
#     proj1 = sample_exponential_power_rff(1.5, d, M)
#     println("射影サンプル例: ", proj1[1:3, :])
#     println()
    
#     # 2. 一般化コーシーカーネル (α=1.5, β=2.0)
#     proj2 = sample_generalized_cauchy_rff(1.5, 2.0, d, M)
#     println("射影サンプル例: ", proj2[1:3, :])
#     println()
    
#     # 3. クンマーカーネル (α=1.5, β=1.5, γ=1.5)
#     proj3 = sample_kummer_rff(1.5, 1.5, 1.5, d, M)
#     println("射影サンプル例: ", proj3[1:3, :])
#     println()
    
#     # 4. ベータカーネル (α=1.5, β=2.0, γ=2.0)
#     proj4 = sample_beta_kernel_rff(1.5, 2.0, 2.0, d, M)
#     println("射影サンプル例: ", proj4[1:3, :])
#     println()
    
#     # 統計確認
#     println("=== 射影統計 ===")
#     for (name, proj) in [("指数べき乗", proj1), ("一般化コーシー", proj2), 
#                         ("クンマー", proj3), ("ベータ", proj4)]
#         println("$name カーネル:")
#         println("  平均: ", round.(mean(proj, dims=1), digits=3))
#         println("  標準偏差: ", round.(std(proj, dims=1), digits=3))
#         println()
#     end
    
#     # RFFカーネル値の計算例
#     println("=== RFFカーネル値例 ===")
#     x1 = [0.0, 0.0]
#     x2 = [1.0, 1.0]
    
#     # 単純な計算例
#     for (name, proj) in [("指数べき乗", proj1), ("一般化コーシー", proj2)]
#         diff = x1 - x2
#         cosine_values = [cos(dot(proj[m, :], diff)) for m in 1:min(100, M)]
#         kernel_value = mean(cosine_values)
#         println("$name: K([0,0], [1,1]) ≈ ", round(kernel_value, digits=4))
#     end
# end

# # 実行
# test_kernels()