#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <vector>

#include "backends/ops.hpp"

namespace py = pybind11;

namespace {

torch::Tensor make_out_like(const torch::Tensor& in) {
    return torch::empty_like(in);
}

torch::Tensor make_equi_linear_out(const torch::Tensor& x, const torch::Tensor& weight) {
    if (x.sizes().size() < 2 || weight.sizes().size() < 2)
        return torch::empty_like(x);

    auto shape              = x.sizes().vec();
    shape[shape.size() - 2] = weight.size(0);
    return torch::empty(shape, x.options());
}

template <typename Fn>
auto make_gp_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor a, torch::Tensor b) {
        RECORD_FUNCTION(name, {});
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        if (a_contig.numel() != b_contig.numel())
            throw std::runtime_error("size mismatch between a and b");
        if (a_contig.size(-1) != 16)
            throw std::runtime_error("last dimension must be 16");
        auto out = make_out_like(a_contig);
        size_t n = a_contig.numel() / 16;
        fn(a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), out.data_ptr<float>(), n);
        return out;
    };
}

template <typename Fn>
auto make_equi_join_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor a, torch::Tensor b, torch::Tensor ref) {
        RECORD_FUNCTION(name, {});
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        if (a_contig.numel() != b_contig.numel())
            throw std::runtime_error("size mismatch between a and b");
        if (a_contig.size(-1) != 16)
            throw std::runtime_error("last dimension must be 16");
        auto out             = make_out_like(a_contig);
        size_t n             = a_contig.numel() / 16;
        const float* ref_ptr = nullptr;
        size_t ref_group     = 1;
        torch::Tensor ref_contig;
        if (ref.defined()) {
            int64_t ref_batch = ref.size(0);
            if (ref.numel() == ref_batch * 16 && n % (size_t)ref_batch == 0) {
                ref_contig = ref.reshape({ref_batch * 16}).contiguous();
                ref_group  = n / (size_t)ref_batch;
            } else {
                ref_contig = ref.expand_as(a_contig).contiguous();
                ref_group  = 1;
            }
            ref_ptr = ref_contig.data_ptr<float>();
        }
        fn(a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), ref_ptr, out.data_ptr<float>(), n, ref_group);
        return out;
    };
}

template <typename Fn>
auto make_equi_linear_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor x, torch::Tensor weight, py::object bias, bool normalize_basis) {
        RECORD_FUNCTION(name, {});
        x        = x.contiguous();
        weight   = weight.contiguous();
        auto out = make_equi_linear_out(x, weight);
        torch::Tensor bias_tensor;
        const float* bias_ptr = nullptr;
        if (!bias.is_none()) {
            bias_tensor = bias.cast<torch::Tensor>().contiguous();
            if (bias_tensor.numel())
                bias_ptr = bias_tensor.data_ptr<float>();
        }
        const auto x_sizes  = x.sizes();
        size_t in_channels  = x_sizes[x_sizes.size() - 2];
        size_t out_channels = weight.size(0);
        size_t batch        = 1;
        for (size_t i = 0; i + 2 < x_sizes.size(); ++i)
            batch *= x_sizes[i];
        fn(x.data_ptr<float>(),
           weight.data_ptr<float>(),
           bias_ptr,
           out.data_ptr<float>(),
           batch,
           in_channels,
           out_channels,
           normalize_basis);
        return out;
    };
}

template <typename Fn>
auto make_equi_rms_norm_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor x, py::object weight, py::object eps) {
        RECORD_FUNCTION(name, {});
        x                       = x.contiguous();
        auto out                = make_out_like(x);
        const float* weight_ptr = nullptr;
        torch::Tensor weight_tensor;
        if (!weight.is_none()) {
            weight_tensor = weight.cast<torch::Tensor>().contiguous();
            if (weight_tensor.numel())
                weight_ptr = weight_tensor.data_ptr<float>();
        }
        float eps_val = 1.1920929e-07f;
        if (!eps.is_none())
            eps_val = eps.cast<float>();
        const auto& sz    = x.sizes();
        size_t n_channels = sz[sz.size() - 2];
        size_t batch      = 1;
        for (size_t i = 0; i + 2 < sz.size(); ++i)
            batch *= sz[i];
        fn(x.data_ptr<float>(), weight_ptr, out.data_ptr<float>(), batch, n_channels, eps_val);
        return out;
    };
}

template <typename Fn>
auto make_scaler_gated_gelu_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor x, py::object approximate) {
        RECORD_FUNCTION(name, {});
        (void)approximate;
        auto x_contig = x.contiguous();
        auto out      = make_out_like(x_contig);
        fn(x_contig.data_ptr<float>(), out.data_ptr<float>(), x_contig.numel());
        return out;
    };
}

template <typename Fn>
auto make_attn_lambda(const char* name, Fn fn) {
    return [name, fn](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs) {
        RECORD_FUNCTION(name, {});
        auto inner_contig = [](const torch::Tensor& t) {
            auto nd = t.dim();
            return t.stride(nd - 1) == 1 && t.stride(nd - 2) == 16;
        };
        bool same_outer = q.stride(0) == k.stride(0) && q.stride(1) == k.stride(1) && q.stride(2) == k.stride(2) &&
                          q.stride(0) == v.stride(0) && q.stride(1) == v.stride(1) && q.stride(2) == v.stride(2);
        if (!(inner_contig(q) && inner_contig(k) && inner_contig(v) && same_outer)) {
            q = q.contiguous();
            k = k.contiguous();
            v = v.contiguous();
        }
        auto sz   = q.sizes();
        int64_t B = sz[0], H = sz[1], T = sz[2], C = sz[3];
        int64_t qs_b = q.stride(0), qs_h = q.stride(1), qs_t = q.stride(2);
        std::vector<std::string> kinds;
        if (kwargs.contains("kinds"))
            for (auto item : kwargs["kinds"].cast<py::dict>())
                kinds.push_back(item.first.cast<std::string>());
        std::vector<torch::Tensor> weight_storage;
        std::vector<const float*> weights;
        if (kwargs.contains("weight"))
            for (auto w : kwargs["weight"].cast<py::list>()) {
                auto wt = w.cast<torch::Tensor>().contiguous();
                weight_storage.push_back(wt);
                weights.push_back(wt.data_ptr<float>());
            }
        torch::Tensor mask_storage;
        const float* mask_ptr = nullptr;
        if (kwargs.contains("attn_mask") && !kwargs["attn_mask"].is_none()) {
            mask_storage = kwargs["attn_mask"].cast<torch::Tensor>().contiguous();
            mask_ptr     = mask_storage.data_ptr<float>();
        }
        bool is_causal = kwargs.contains("is_causal") && kwargs["is_causal"].cast<bool>();
        auto out       = torch::empty({B, H, T, C, 16}, v.options());
        fn(q.data_ptr<float>(),
           k.data_ptr<float>(),
           v.data_ptr<float>(),
           out.data_ptr<float>(),
           B,
           H,
           T,
           C,
           qs_b,
           qs_h,
           qs_t,
           kinds,
           weights,
           mask_ptr,
           is_causal);
        return py::make_tuple(out, py::none());
    };
}

}  // namespace

PYBIND11_MODULE(turbogator_ext, m) {
    // geometric_product
    m.def("geometric_product_baseline",
          make_gp_lambda("turbogator::geometric_product_baseline", turbogator::geometric_product_baseline));
    m.def("geometric_product_opt_v1",
          make_gp_lambda("turbogator::geometric_product_opt_v1", turbogator::geometric_product_opt_v1));
    m.def("geometric_product_vectorized",
          make_gp_lambda("turbogator::geometric_product_vectorized", turbogator::geometric_product_vectorized));

    // equi_join
    m.def("equi_join_baseline",
          make_equi_join_lambda("turbogator::equi_join_baseline", turbogator::equi_join_baseline));
    m.def("equi_join_opt_v1", make_equi_join_lambda("turbogator::equi_join_opt_v1", turbogator::equi_join_opt_v1));
    m.def("equi_join_opt_v2", make_equi_join_lambda("turbogator::equi_join_opt_v2", turbogator::equi_join_opt_v2));
    m.def("equi_join_vectorized",
          make_equi_join_lambda("turbogator::equi_join_vectorized", turbogator::equi_join_vectorized));

    // equi_geometric_attention
    m.def("equi_geometric_attention_baseline",
          make_attn_lambda("turbogator::equi_geometric_attention_baseline",
                           turbogator::equi_geometric_attention_baseline));
    m.def("equi_geometric_attention_opt_v1",
          make_attn_lambda("turbogator::equi_geometric_attention_opt_v1", turbogator::equi_geometric_attention_opt_v1));
    m.def("equi_geometric_attention_vectorized",
          make_attn_lambda("turbogator::equi_geometric_attention_vectorized",
                           turbogator::equi_geometric_attention_vectorized));

    // scaler_gated_gelu
    m.def(
        "scaler_gated_gelu_baseline",
        make_scaler_gated_gelu_lambda("turbogator::scaler_gated_gelu_baseline", turbogator::scaler_gated_gelu_baseline),
        py::arg("x"),
        py::arg("approximate") = "tanh");
    m.def("scaler_gated_gelu_vectorized",
          make_scaler_gated_gelu_lambda("turbogator::scaler_gated_gelu_vectorized",
                                        turbogator::scaler_gated_gelu_vectorized),
          py::arg("x"),
          py::arg("approximate") = "tanh");

    // equi_linear
    m.def("equi_linear_baseline",
          make_equi_linear_lambda("turbogator::equi_linear_baseline", turbogator::equi_linear_baseline),
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias")            = py::none(),
          py::arg("normalize_basis") = true);
    m.def("equi_linear_opt_v1",
          make_equi_linear_lambda("turbogator::equi_linear_opt_v1", turbogator::equi_linear_opt_v1),
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias")            = py::none(),
          py::arg("normalize_basis") = true);
    m.def("equi_linear_opt_v2",
          make_equi_linear_lambda("turbogator::equi_linear_opt_v2", turbogator::equi_linear_opt_v2),
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias")            = py::none(),
          py::arg("normalize_basis") = true);
    m.def("equi_linear_vectorized",
          make_equi_linear_lambda("turbogator::equi_linear_vectorized", turbogator::equi_linear_vectorized),
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias")            = py::none(),
          py::arg("normalize_basis") = true);

    // equi_rms_norm
    m.def("equi_rms_norm_baseline",
          make_equi_rms_norm_lambda("turbogator::equi_rms_norm_baseline", turbogator::equi_rms_norm_baseline),
          py::arg("x"),
          py::arg("weight") = py::none(),
          py::arg("eps")    = py::none());
    m.def("equi_rms_norm_opt_v1",
          make_equi_rms_norm_lambda("turbogator::equi_rms_norm_opt_v1", turbogator::equi_rms_norm_opt_v1),
          py::arg("x"),
          py::arg("weight") = py::none(),
          py::arg("eps")    = py::none());
    m.def("equi_rms_norm_vectorized",
          make_equi_rms_norm_lambda("turbogator::equi_rms_norm_vectorized", turbogator::equi_rms_norm_vectorized),
          py::arg("x"),
          py::arg("weight") = py::none(),
          py::arg("eps")    = py::none());
}
