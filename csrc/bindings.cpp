#include <malloc.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <type_traits>
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

static std::tuple<size_t, size_t, size_t> detect_outer_strides(const torch::Tensor& a, const torch::Tensor& b) {
    auto none = std::make_tuple(size_t(0), size_t(0), size_t(0));
    if (a.is_contiguous() && b.is_contiguous())
        return none;
    int nda = a.dim(), ndb = b.dim();
    // Need (channel, blade) inner dims plus a single outer dim
    if (nda < 3 || ndb < 3 || nda > 4 || ndb > 4)
        return none;
    if (a.stride(nda - 1) != 1 || a.stride(nda - 2) != 16)
        return none;
    if (b.stride(ndb - 1) != 1 || b.stride(ndb - 2) != 16)
        return none;
    int64_t bs = a.size(nda - 2);
    if (bs != b.size(ndb - 2) || bs == 0)
        return none;
    int64_t os_a = a.stride(nda - 3), os_b = b.stride(ndb - 3);
    if (os_a % (bs * 16) != 0 || os_b % (bs * 16) != 0)
        return none;
    if (nda == 4 && a.stride(0) != a.size(1) * a.stride(1))
        return none;
    if (ndb == 4 && b.stride(0) != b.size(1) * b.stride(1))
        return none;
    return std::make_tuple(size_t(bs), size_t(os_a), size_t(os_b));
}

template <typename Fn>
auto make_gp_lambda(const char* name, Fn fn) {
    constexpr bool kStrided =
        std::is_invocable_v<Fn, const float*, const float*, float*, size_t, size_t, size_t, size_t>;
    return [name, fn](torch::Tensor a, torch::Tensor b) {
        RECORD_FUNCTION(name, {});
        if (a.numel() != b.numel())
            throw std::runtime_error("size mismatch between a and b");
        if (a.size(-1) != 16)
            throw std::runtime_error("last dimension must be 16");
        size_t n = a.numel() / 16;
        if constexpr (kStrided) {
            auto [bs, os_a, os_b] = detect_outer_strides(a, b);
            if (bs > 0) {
                auto out = torch::empty(a.sizes().vec(), a.options());
                fn(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, bs, os_a, os_b);
                return out;
            }
        }
        auto a_contig = a.contiguous();
        auto b_contig = b.contiguous();
        auto out      = make_out_like(a_contig);
        if constexpr (kStrided)
            fn(a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), out.data_ptr<float>(), n, 0, 0, 0);
        else
            fn(a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), out.data_ptr<float>(), n);
        return out;
    };
}

static const float* resolve_ref(
    const torch::Tensor& ref, const torch::Tensor& a, size_t n, torch::Tensor& ref_contig, size_t& ref_group) {
    ref_group = 1;
    if (!ref.defined())
        return nullptr;
    int64_t ref_batch = ref.size(0);
    if (ref.numel() == ref_batch * 16 && n % (size_t)ref_batch == 0) {
        ref_contig = ref.reshape({ref_batch * 16}).contiguous();
        ref_group  = n / (size_t)ref_batch;
    } else {
        ref_contig = ref.expand_as(a).contiguous();
        ref_group  = 1;
    }
    return ref_contig.data_ptr<float>();
}

// avoid strided pytorch tensors
// true -> directly use mem
// false -> copy first
static bool io_strides(const torch::Tensor& a,
                       const torch::Tensor& b,
                       const torch::Tensor& out,
                       size_t& bs,
                       size_t& os_a,
                       size_t& os_b,
                       size_t& os_out) {
    auto inner = [](const torch::Tensor& t) {
        int nd = t.dim();
        return nd >= 3 && nd <= 4 && t.stride(nd - 1) == 1 && t.stride(nd - 2) == 16;
    };
    if (!inner(a) || !inner(b) || !inner(out))
        return false;
    int64_t c = a.size(a.dim() - 2);
    if (c <= 0 || b.size(b.dim() - 2) != c || out.size(out.dim() - 2) != c)
        return false;
    auto mergeok = [](const torch::Tensor& t) { return t.dim() != 4 || t.stride(0) == t.size(1) * t.stride(1); };
    if (!mergeok(a) || !mergeok(b) || !mergeok(out))
        return false;
    bs     = (size_t)c;
    os_a   = (size_t)a.stride(a.dim() - 3);
    os_b   = (size_t)b.stride(b.dim() - 3);
    os_out = (size_t)out.stride(out.dim() - 3);
    return true;
}

// gp variant to write directly into buff
static void gp_vectorized_out(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    RECORD_FUNCTION("turbogator::geometric_product_vectorized_out", {});
    if (a.numel() != b.numel() || a.numel() != out.numel())
        throw std::runtime_error("size mismatch (a/b/out)");
    if (a.size(-1) != 16)
        throw std::runtime_error("last dimension must be 16");
    size_t n = a.numel() / 16;
    size_t bs, os_a, os_b, os_out;
    if (io_strides(a, b, out, bs, os_a, os_b, os_out)) {
        turbogator::geometric_product_vectorized_out(
            a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, bs, os_a, os_b, os_out);
    } else {
        auto ac = a.contiguous(), bc = b.contiguous();
        auto tmp = torch::empty(ac.sizes().vec(), ac.options());
        turbogator::geometric_product_vectorized(
            ac.data_ptr<float>(), bc.data_ptr<float>(), tmp.data_ptr<float>(), n, 0, 0, 0);
        out.copy_(tmp);
    }
}

// join variant to write directly into buff
static void join_vectorized_out(torch::Tensor a, torch::Tensor b, torch::Tensor ref, torch::Tensor out) {
    RECORD_FUNCTION("turbogator::equi_join_vectorized_out", {});
    if (a.numel() != b.numel() || a.numel() != out.numel())
        throw std::runtime_error("size mismatch (a/b/out)");
    if (a.size(-1) != 16)
        throw std::runtime_error("last dimension must be 16");
    size_t n         = a.numel() / 16;
    size_t ref_group = 1;
    torch::Tensor ref_contig;
    size_t bs, os_a, os_b, os_out;
    if (io_strides(a, b, out, bs, os_a, os_b, os_out)) {
        const float* ref_ptr = resolve_ref(ref, a, n, ref_contig, ref_group);
        turbogator::equi_join_vectorized_out(a.data_ptr<float>(),
                                             b.data_ptr<float>(),
                                             ref_ptr,
                                             out.data_ptr<float>(),
                                             n,
                                             ref_group,
                                             bs,
                                             os_a,
                                             os_b,
                                             os_out);
    } else {
        auto ac = a.contiguous(), bc = b.contiguous();
        auto tmp             = torch::empty(ac.sizes().vec(), ac.options());
        const float* ref_ptr = resolve_ref(ref, ac, n, ref_contig, ref_group);
        turbogator::equi_join_vectorized(
            ac.data_ptr<float>(), bc.data_ptr<float>(), ref_ptr, tmp.data_ptr<float>(), n, ref_group);
        out.copy_(tmp);
    }
}

template <typename Fn>
auto make_equi_join_lambda(const char* name, Fn fn) {
    constexpr bool kStrided = std::
        is_invocable_v<Fn, const float*, const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t>;
    return [name, fn](torch::Tensor a, torch::Tensor b, torch::Tensor ref) {
        RECORD_FUNCTION(name, {});
        if (a.numel() != b.numel())
            throw std::runtime_error("size mismatch between a and b");
        if (a.size(-1) != 16)
            throw std::runtime_error("last dimension must be 16");
        size_t n         = a.numel() / 16;
        size_t ref_group = 1;
        torch::Tensor ref_contig;
        if constexpr (kStrided) {
            const float* ref_ptr  = resolve_ref(ref, a, n, ref_contig, ref_group);
            auto [bs, os_a, os_b] = detect_outer_strides(a, b);
            if (bs > 0) {
                auto out = torch::empty(a.sizes().vec(), a.options());
                fn(a.data_ptr<float>(),
                   b.data_ptr<float>(),
                   ref_ptr,
                   out.data_ptr<float>(),
                   n,
                   ref_group,
                   bs,
                   os_a,
                   os_b);
                return out;
            }
        }
        auto a_contig        = a.contiguous();
        auto b_contig        = b.contiguous();
        auto out             = make_out_like(a_contig);
        const float* ref_ptr = resolve_ref(ref, a_contig, n, ref_contig, ref_group);
        if constexpr (kStrided)
            fn(a_contig.data_ptr<float>(),
               b_contig.data_ptr<float>(),
               ref_ptr,
               out.data_ptr<float>(),
               n,
               ref_group,
               0,
               0,
               0);
        else
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
auto make_attn_lambda(const char* name, Fn fn, bool transposed_out = false) {
    return [name, fn, transposed_out](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs) {
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

        // kernel emit (B, T, H*C, 16) directly so next equi_linear can use it without rearrange/copy
        auto out =
            transposed_out ? torch::empty({B, T, H * C, 16}, v.options()) : torch::empty({B, H, T, C, 16}, v.options());
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
    // stop page faults from allocate on touch policy
    // alloc from retained heap, never return the heap to the os
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);

    // geometric_product
    m.def("geometric_product_baseline",
          make_gp_lambda("turbogator::geometric_product_baseline", turbogator::geometric_product_baseline));
    m.def("geometric_product_opt_v1",
          make_gp_lambda("turbogator::geometric_product_opt_v1", turbogator::geometric_product_opt_v1));
    m.def("geometric_product_vectorized",
          make_gp_lambda("turbogator::geometric_product_vectorized", turbogator::geometric_product_vectorized));
    m.def("geometric_product_vectorized_out", &gp_vectorized_out);

    // equi_join
    m.def("equi_join_baseline",
          make_equi_join_lambda("turbogator::equi_join_baseline", turbogator::equi_join_baseline));
    m.def("equi_join_opt_v1", make_equi_join_lambda("turbogator::equi_join_opt_v1", turbogator::equi_join_opt_v1));
    m.def("equi_join_opt_v2", make_equi_join_lambda("turbogator::equi_join_opt_v2", turbogator::equi_join_opt_v2));
    m.def("equi_join_vectorized",
          make_equi_join_lambda("turbogator::equi_join_vectorized", turbogator::equi_join_vectorized));
    m.def("equi_join_vectorized_out", &join_vectorized_out);

    // equi_geometric_attention
    m.def("equi_geometric_attention_baseline",
          make_attn_lambda("turbogator::equi_geometric_attention_baseline",
                           turbogator::equi_geometric_attention_baseline));
    m.def("equi_geometric_attention_opt_v1",
          make_attn_lambda("turbogator::equi_geometric_attention_opt_v1", turbogator::equi_geometric_attention_opt_v1));
    m.def("equi_geometric_attention_opt_v2",
          make_attn_lambda("turbogator::equi_geometric_attention_opt_v2", turbogator::equi_geometric_attention_opt_v2));
    m.def(
        "equi_geometric_attention_vectorized",
        make_attn_lambda(
            "turbogator::equi_geometric_attention_vectorized", turbogator::equi_geometric_attention_vectorized, true));

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
