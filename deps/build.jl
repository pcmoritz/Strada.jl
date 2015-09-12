using BinDeps
using Compat

@osx? if true using Homebrew; end : nothing

@BinDeps.setup

glog = library_dependency("libglog")
probobuf = library_dependency("libprotobuf")

libjlcaffe = library_dependency("libjlcaffe")

stradasrcdir = joinpath(Pkg.dir("Strada"), "src")
prefix = joinpath(BinDeps.depsdir(libjlcaffe), "usr")
protobufdir = joinpath(Pkg.dir("ProtoBuf"), "plugin")
builddir = joinpath(BinDeps.depsdir(libjlcaffe), "build")
srcdir = joinpath(BinDeps.depsdir(libjlcaffe), "src")

ENV["PATH"] = ENV["PATH"] * ":" * protobufdir

provides(AptGet,{
        "libgoogle-glog-dev" => glog,
        "libprotobuf-dev" => probobuf,
    })

@osx ? begin
provides(Homebrew.HB,
    "glog", glog,
    "protobuf241", protobuf,
    os = :Darwin)
end : nothing

provides(SimpleBuild,
    (@build_steps begin
        CreateDirectory(builddir)
        CreateDirectory(joinpath(prefix, "lib"))
        @build_steps begin
            ChangeDirectory(builddir)
            FileRule(joinpath(prefix,"lib","libjlcaffe.so"),@build_steps begin
                `sudo apt-get install -y libboost-system-dev libboost-thread-dev protobuf-compiler cmake`
                if Base.blas_vendor() == :openblas64
                    `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DUSE_64_BIT_BLAS=ON $srcdir`
                else
                    `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" $srcdir`
                end
                `make -j`
                `cp libjlcaffe.so $prefix/lib`
                `protoc -I=$srcdir/caffe/src/caffe/proto/ --julia_out=$stradasrcdir/ $srcdir/caffe/src/caffe/proto/caffe.proto`
            end)
        end
    end), libjlcaffe, os=:Unix, installed_libpath=joinpath(prefix, "lib"))

provides(SimpleBuild,
    (@build_steps begin
        CreateDirectory(builddir)
        CreateDirectory(joinpath(prefix, "lib"))
        @build_steps begin
            ChangeDirectory(builddir)
            FileRule(joinpath(prefix,"lib","libjlcaffe.so"),@build_steps begin
                `brew install boost cmake`
                if Base.blas_vendor() == :openblas64
                    `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DUSE_64_BIT_BLAS=ON $srcdir`
                else
                    `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" $srcdir`
                end
                `make -j`
                `cp libjlcaffe.so $prefix/lib`
                `protoc -I=$srcdir/caffe/src/caffe/proto/ --julia_out=$stradasrcdir/ $srcdir/caffe/src/caffe/proto/caffe.proto`
            end)
        end
    end), libjlcaffe, os=:Darwin, installed_libpath=joinpath(prefix, "lib"))

@BinDeps.install @compat Dict(:libjlcaffe => :libjlc)
