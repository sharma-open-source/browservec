(module
  (memory (export "mem") 1)

  ;; dot(dataPtr, qPtr, outPtr, rows, dim) — pointers are BYTE offsets.
  ;; out[r] = sum_i data[r*dim+i] * q[i].  Used for cosine (pre-normalized) and dot.
  (func (export "dot")
    (param $data i32) (param $q i32) (param $out i32) (param $rows i32) (param $dim i32)
    (local $r i32) (local $base i32) (local $i i32) (local $tail i32)
    (local $acc v128) (local $sum f32) (local $dimBytes i32)
    (local.set $tail (i32.and (local.get $dim) (i32.const -4)))
    (local.set $dimBytes (i32.mul (local.get $dim) (i32.const 4)))
    (local.set $r (i32.const 0))
    (block $rdone (loop $rloop
      (br_if $rdone (i32.ge_s (local.get $r) (local.get $rows)))
      (local.set $base (i32.add (local.get $data) (i32.mul (local.get $r) (local.get $dimBytes))))
      (local.set $acc (v128.const i32x4 0 0 0 0))
      (local.set $i (i32.const 0))
      (block $vdone (loop $vloop
        (br_if $vdone (i32.ge_s (local.get $i) (local.get $tail)))
        (local.set $acc
          (f32x4.add (local.get $acc)
            (f32x4.mul
              (v128.load (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4))))
              (v128.load (i32.add (local.get $q)    (i32.mul (local.get $i) (i32.const 4)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $vloop)))
      (local.set $sum
        (f32.add
          (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
          (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))))
      (block $tdone (loop $tloop
        (br_if $tdone (i32.ge_s (local.get $i) (local.get $dim)))
        (local.set $sum
          (f32.add (local.get $sum)
            (f32.mul
              (f32.load (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4))))
              (f32.load (i32.add (local.get $q)    (i32.mul (local.get $i) (i32.const 4)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tloop)))
      (f32.store (i32.add (local.get $out) (i32.mul (local.get $r) (i32.const 4))) (local.get $sum))
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br $rloop)))
  )

  ;; l2(dataPtr, qPtr, outPtr, rows, dim) — out[r] = -sum_i (data-q)^2  (higher = closer).
  (func (export "l2")
    (param $data i32) (param $q i32) (param $out i32) (param $rows i32) (param $dim i32)
    (local $r i32) (local $base i32) (local $i i32) (local $tail i32)
    (local $acc v128) (local $diff v128) (local $sum f32) (local $d f32) (local $dimBytes i32)
    (local.set $tail (i32.and (local.get $dim) (i32.const -4)))
    (local.set $dimBytes (i32.mul (local.get $dim) (i32.const 4)))
    (local.set $r (i32.const 0))
    (block $rdone (loop $rloop
      (br_if $rdone (i32.ge_s (local.get $r) (local.get $rows)))
      (local.set $base (i32.add (local.get $data) (i32.mul (local.get $r) (local.get $dimBytes))))
      (local.set $acc (v128.const i32x4 0 0 0 0))
      (local.set $i (i32.const 0))
      (block $vdone (loop $vloop
        (br_if $vdone (i32.ge_s (local.get $i) (local.get $tail)))
        (local.set $diff
          (f32x4.sub
            (v128.load (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4))))
            (v128.load (i32.add (local.get $q)    (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $acc (f32x4.add (local.get $acc) (f32x4.mul (local.get $diff) (local.get $diff))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $vloop)))
      (local.set $sum
        (f32.add
          (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
          (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))))
      (block $tdone (loop $tloop
        (br_if $tdone (i32.ge_s (local.get $i) (local.get $dim)))
        (local.set $d
          (f32.sub
            (f32.load (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4))))
            (f32.load (i32.add (local.get $q)    (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $sum (f32.add (local.get $sum) (f32.mul (local.get $d) (local.get $d))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tloop)))
      (f32.store (i32.add (local.get $out) (i32.mul (local.get $r) (i32.const 4))) (f32.neg (local.get $sum)))
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br $rloop)))
  )
)
