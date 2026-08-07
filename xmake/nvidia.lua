-- CUDA support. Included only when --nv-gpu=y is set (see xmake.lua).
-- The .cu kernels are compiled into the final shared library target directly
-- (llaisys target adds these files when nv-gpu is on), so xmake drives nvcc
-- for both compilation and the device-link step.

set_policy("check.auto_ignore_flags", false)

-- Force dynamic CRT (/MD) so nvcc-compiled objects match MSVC-compiled ones
-- (avoids LNK2038 RuntimeLibrary mismatch MT vs MD).
local nv_md = "-Xcompiler=/MD"

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        add_cxflags("/utf-8", {force = true})
        add_cuflags("-Xcompiler=/utf-8", {force = true})
        add_cxflags("/wd4819", {force = true})
        add_cuflags("-Xcompiler=/wd4819", {force = true})
        add_cuflags(nv_md, {force = true})
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        add_cxflags("/utf-8", {force = true})
        add_cuflags("-Xcompiler=/utf-8", {force = true})
        add_cxflags("/wd4819", {force = true})
        add_cuflags("-Xcompiler=/wd4819", {force = true})
        add_cuflags(nv_md, {force = true})
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
