#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <vector>

#include "backends/ops.hpp"

namespace py = pybind11;

namespace
{

    torch::Tensor make_out_like(const torch::Tensor &in)
    {
        return torch::empty_like(in);
    }

    torch::Tensor make_equi_linear_out(const torch::Tensor &x, const torch::Tensor &weight)
    {

        if (x.sizes().size() < 2 || weight.sizes().size() < 2)
        {
            return torch::empty_like(x);
        }

        auto shape = x.sizes().vec();
        shape[shape.size() - 2] = weight.size(0);
        return torch::empty(shape, x.options());
    }

}

PYBIND11_MODULE(turbogator_ext, m)
{
    m.def("geometric_product_baseline", [](torch::Tensor a, torch::Tensor b)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        if (a_contig.numel() != b_contig.numel())
        {
            throw std::runtime_error("geometric_product_baseline: size mismatch between a and b");
        }

        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("geometric_product_baseline: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);

        size_t num_multivectors = a_contig.numel() / 16;

        turbogator::geometric_product_baseline(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });

    m.def("geometric_product_opt_v1", [](torch::Tensor a, torch::Tensor b)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        if (a_contig.numel() != b_contig.numel())
        {
            throw std::runtime_error("geometric_product_opt_v1: size mismatch between a and b");
        }

        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("geometric_product_opt_v1: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);

        size_t num_multivectors = a_contig.numel() / 16;

        turbogator::geometric_product_opt_v1(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            num_multivectors
        );

        return out; });

    m.def("geometric_product_vectorized", [](torch::Tensor a, torch::Tensor b)
          {
        if (a.numel() != b.numel()) {
            throw std::runtime_error("geometric_product_vectorized: size mismatch");
        }
        auto out = make_out_like(a);
        turbogator::geometric_product_vectorized(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            a.numel()
        );


        return    out; });

    m.def("equi_join_baseline", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_baseline: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_baseline: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_baseline(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });
    
    m.def("equi_join_optimized_hardcoded", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_optimized_hardcoded: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_optimized_hardcoded: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_optimized_hardcoded(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });

    m.def("equi_join_optimized_sparse", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_optimized_sparse: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_optimized_sparse: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_optimized_sparse(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });
    
    m.def("equi_join_optimized_precompute_ab", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_optimized_precompute_ab: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_optimized_precompute_ab: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_optimized_precompute_ab(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });

    m.def("equi_join_optimized_unroll_k", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_optimized_unroll_k: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_optimized_unroll_k: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_optimized_unroll_k(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });
    
    m.def("equi_join_restrict_unswitch", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref)
          {
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();

        if (a_contig.numel() != b_contig.numel()) {
            throw std::runtime_error("equi_join_restrict_unswitch: size mismatch between a and b");
        }
        
        if (a_contig.size(-1) != 16) {
            throw std::runtime_error("equi_join_restrict_unswitch: last dimension must be 16");
        }

        auto out = make_out_like(a_contig);
        size_t num_multivectors = a_contig.numel() / 16;

        const float* ref_ptr = nullptr;
        torch::Tensor ref_contig; 
        if (ref.defined()) {
            ref_contig = ref.expand_as(a_contig).contiguous();
            ref_ptr = ref_contig.data_ptr<float>();
        }

        turbogator::equi_join_restrict_unswitch(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            ref_ptr, 
            out.data_ptr<float>(),
            num_multivectors
        );
        
        return out; });


    m.def(
        "equi_geometric_attention_baseline",
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs)
        {
            // q, k, v: (B, H, T, C, 16)
            q = q.contiguous();
            k = k.contiguous();
            v = v.contiguous();
            auto sz = q.sizes();
            int64_t B = sz[0], H = sz[1], T = sz[2], C = sz[3];

            std::vector<std::string> kinds;
            if (kwargs.contains("kinds"))
                for (auto item : kwargs["kinds"].cast<py::dict>())
                    kinds.push_back(item.first.cast<std::string>());

            // weights: one per kind, shape (H, 1, C, 1) -> H*C floats, w[h*C + c]
            std::vector<torch::Tensor> weight_storage;
            std::vector<const float *> weights;
            if (kwargs.contains("weight"))
                for (auto w : kwargs["weight"].cast<py::list>())
                {
                    auto wt = w.cast<torch::Tensor>().contiguous();
                    weight_storage.push_back(wt);
                    weights.push_back(wt.data_ptr<float>());
                }

            torch::Tensor mask_storage;
            const float *mask_ptr = nullptr;
            if (kwargs.contains("attn_mask") && !kwargs["attn_mask"].is_none())
            {
                mask_storage = kwargs["attn_mask"].cast<torch::Tensor>().contiguous();
                mask_ptr = mask_storage.data_ptr<float>();
            }

            bool is_causal = kwargs.contains("is_causal") && kwargs["is_causal"].cast<bool>();

            auto out = torch::empty_like(v);
            turbogator::equi_geometric_attention_baseline(
                q.data_ptr<float>(), k.data_ptr<float>(),
                v.data_ptr<float>(), out.data_ptr<float>(),
                B, H, T, C, kinds, weights, mask_ptr, is_causal);
            return py::make_tuple(out, py::none());
        });
    m.def(
        "equi_geometric_attention_optimized1",
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs)
        {
            // q, k, v: (B, H, T, C, 16)
            q = q.contiguous();
            k = k.contiguous();
            v = v.contiguous();
            auto sz = q.sizes();
            int64_t B = sz[0], H = sz[1], T = sz[2], C = sz[3];

            std::vector<std::string> kinds;
            if (kwargs.contains("kinds"))
                for (auto item : kwargs["kinds"].cast<py::dict>())
                    kinds.push_back(item.first.cast<std::string>());

            // weights: one per kind, shape (H, 1, C, 1) -> H*C floats, w[h*C + c]
            std::vector<torch::Tensor> weight_storage;
            std::vector<const float *> weights;
            if (kwargs.contains("weight"))
                for (auto w : kwargs["weight"].cast<py::list>())
                {
                    auto wt = w.cast<torch::Tensor>().contiguous();
                    weight_storage.push_back(wt);
                    weights.push_back(wt.data_ptr<float>());
                }

            torch::Tensor mask_storage;
            const float *mask_ptr = nullptr;
            if (kwargs.contains("attn_mask") && !kwargs["attn_mask"].is_none())
            {
                mask_storage = kwargs["attn_mask"].cast<torch::Tensor>().contiguous();
                mask_ptr = mask_storage.data_ptr<float>();
            }

            bool is_causal = kwargs.contains("is_causal") && kwargs["is_causal"].cast<bool>();

            auto out = torch::empty_like(v);
            turbogator::equi_geometric_attention_optimized1(
                q.data_ptr<float>(), k.data_ptr<float>(),
                v.data_ptr<float>(), out.data_ptr<float>(),
                B, H, T, C, kinds, weights, mask_ptr, is_causal);
            return py::make_tuple(out, py::none());
        });
    m.def(
        "equi_geometric_attention_optimized2",
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs)
        {
            // q, k, v: (B, H, T, C, 16)
            q = q.contiguous();
            k = k.contiguous();
            v = v.contiguous();
            auto sz = q.sizes();
            int64_t B = sz[0], H = sz[1], T = sz[2], C = sz[3];

            std::vector<std::string> kinds;
            if (kwargs.contains("kinds"))
                for (auto item : kwargs["kinds"].cast<py::dict>())
                    kinds.push_back(item.first.cast<std::string>());

            std::vector<torch::Tensor> weight_storage;
            std::vector<const float *> weights;
            if (kwargs.contains("weight"))
                for (auto w : kwargs["weight"].cast<py::list>())
                {
                    auto wt = w.cast<torch::Tensor>().contiguous();
                    weight_storage.push_back(wt);
                    weights.push_back(wt.data_ptr<float>());
                }

            torch::Tensor mask_storage;
            const float *mask_ptr = nullptr;
            if (kwargs.contains("attn_mask") && !kwargs["attn_mask"].is_none())
            {
                mask_storage = kwargs["attn_mask"].cast<torch::Tensor>().contiguous();
                mask_ptr = mask_storage.data_ptr<float>();
            }

            bool is_causal = kwargs.contains("is_causal") && kwargs["is_causal"].cast<bool>();

            auto out = torch::empty_like(v);
            turbogator::equi_geometric_attention_optimized2(
                q.data_ptr<float>(), k.data_ptr<float>(),
                v.data_ptr<float>(), out.data_ptr<float>(),
                B, H, T, C, kinds, weights, mask_ptr, is_causal);
            return py::make_tuple(out, py::none());
        });

    m.def(
        "scaler_gated_gelu_baseline",
        [](torch::Tensor x, py::object approximate)
        {
            (void)approximate;
            auto x_contig = x.contiguous();
            auto out = make_out_like(x_contig);
            turbogator::scaler_gated_gelu_baseline(
                x_contig.data_ptr<float>(),
                out.data_ptr<float>(),
                x_contig.numel());
            return out;
        },
        py::arg("x"),
        py::arg("approximate") = "tanh");

    m.def(
        "equi_linear_baseline",
        [](torch::Tensor x, torch::Tensor weight, py::object bias, bool normalize_basis)
        {
            x = x.contiguous();
            weight = weight.contiguous();
            auto out = make_equi_linear_out(x, weight);

            torch::Tensor bias_tensor;
            const float *bias_ptr = nullptr;
            if (!bias.is_none())
            {
                bias_tensor = bias.cast<torch::Tensor>().contiguous();
                if (bias_tensor.numel())
                {
                    bias_ptr = bias_tensor.data_ptr<float>();
                }
            }

            // x: (..., in_channels, 16); weight: (out_channels, in_channels, 9)
            const auto x_sizes = x.sizes();
            size_t in_channels = x_sizes[x_sizes.size() - 2];
            size_t out_channels = weight.size(0);
            size_t batch = 1;
            for (size_t i = 0; i + 2 < x_sizes.size(); ++i)
                batch *= x_sizes[i];

            turbogator::equi_linear_baseline(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
                out.data_ptr<float>(), batch, in_channels, out_channels, normalize_basis);
            return out;
        },
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("normalize_basis") = true);

    m.def(
        "equi_linear_opt_v2",
        [](torch::Tensor x, torch::Tensor weight, py::object bias, bool normalize_basis)
        {
            x = x.contiguous();
            weight = weight.contiguous();
            auto out = make_equi_linear_out(x, weight);

            torch::Tensor bias_tensor;
            const float *bias_ptr = nullptr;
            if (!bias.is_none())
            {
                bias_tensor = bias.cast<torch::Tensor>().contiguous();
                if (bias_tensor.numel())
                    bias_ptr = bias_tensor.data_ptr<float>();
            }

            const auto x_sizes = x.sizes();
            size_t in_channels = x_sizes[x_sizes.size() - 2];
            size_t out_channels = weight.size(0);
            size_t batch = 1;
            for (size_t i = 0; i + 2 < x_sizes.size(); ++i)
                batch *= x_sizes[i];

            turbogator::equi_linear_opt_v2(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
                out.data_ptr<float>(), batch, in_channels, out_channels, normalize_basis);
            return out;
        },
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("normalize_basis") = true);

    m.def(
        "equi_rms_norm_baseline",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            torch::Tensor weight_tensor;
            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f; // torch.finfo(float32).eps
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            // x: (..., n_channels, 16)
            const auto &sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i)
                batch *= sz[i];

            turbogator::equi_rms_norm_baseline(
                x.data_ptr<float>(), weight_ptr, out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    
    m.def(
        "equi_rms_norm_branchless_clamp",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_branchless_clamp(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());

    m.def(
        "equi_rms_norm_restrict",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_restrict(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());

    m.def(
        "equi_rms_norm_unrolled_selector",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_unrolled_selector(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    
    m.def(
        "equi_rms_norm_reciprocal_div",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_reciprocal_div(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    
    m.def(
        "equi_rms_norm_prefetch",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_prefetch(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());

    m.def(
        "equi_rms_norm_unrolled_channels_4",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_unrolled_channels_4(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    
    m.def(
        "equi_rms_norm_assume_aligned",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_assume_aligned(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    
    m.def(
        "equi_rms_norm_combined",
        [](torch::Tensor x, py::object weight, py::object eps)
        {
            x = x.contiguous();
            auto out = make_out_like(x);

            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>().contiguous();
                if (weight_tensor.numel())
                    weight_ptr = weight_tensor.data_ptr<float>();
            }

            float eps_val = 1.1920929e-07f;
            if (!eps.is_none())
                eps_val = eps.cast<float>();

            const auto& sz = x.sizes();
            size_t n_channels = sz[sz.size() - 2];
            size_t batch = 1;
            for (size_t i = 0; i + 2 < sz.size(); ++i) batch *= sz[i];

            turbogator::equi_rms_norm_combined(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                batch, n_channels, eps_val);
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
}
