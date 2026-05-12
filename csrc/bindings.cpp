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

            // weights: one per kind, shape (H, 1, C, 1) → H*C floats, w[h*C + c]
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
        "scaler_gated_gelu_baseline",
        [](torch::Tensor x, py::object approximate)
        {
            (void)approximate;
            auto out = make_out_like(x);
            turbogator::scaler_gated_gelu_baseline(
                x.data_ptr<float>(),
                out.data_ptr<float>(),
                x.numel());
            return out;
        },
        py::arg("x"),
        py::arg("approximate") = "tanh");

    m.def(
        "equi_linear_baseline",
        [](torch::Tensor x, torch::Tensor weight, py::object bias, bool normalize_basis)
        {
            (void)normalize_basis;
            auto out = make_equi_linear_out(x, weight);
            const float *bias_ptr = nullptr;
            if (!bias.is_none())
            {
                auto bias_tensor = bias.cast<torch::Tensor>();
                if (bias_tensor.numel())
                {
                    bias_ptr = bias_tensor.data_ptr<float>();
                }
            }
            turbogator::equi_linear_baseline(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias_ptr,
                out.data_ptr<float>(),
                out.numel());
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
            (void)eps;
            auto out = make_out_like(x);
            const float *weight_ptr = nullptr;
            if (!weight.is_none())
            {
                auto weight_tensor = weight.cast<torch::Tensor>();
                if (weight_tensor.numel())
                {
                    weight_ptr = weight_tensor.data_ptr<float>();
                }
            }
            turbogator::equi_rms_norm_baseline(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                out.numel());
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
}
