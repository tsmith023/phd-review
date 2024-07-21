use console::Emoji;
use indicatif::ProgressBar;
use nalgebra::{ComplexField, DMatrix};
use num_dual::*;
use rayon::prelude::*;
use rusty_rootsearch::{root_search, RootSearchOptions};
use serde::{Deserialize, Serialize};
use std;

const PI: f32 = std::f32::consts::PI;

#[derive(PartialEq, Debug)]
enum SystemType {
    Bulk,
    Finite,
}

struct KronigPenney {
    u1: f32,
    u2: f32,
    d: f32,
    v: f32,
    // w: f32,
    x1: f32,
    x2: f32,
    n: u32,
    lower: f32,
    upper: f32,
    steps: i32,
    resolution: i32,
    patience: i32,
    tolerance: f32,
}

#[derive(Serialize, Deserialize)]
struct RootsPerIterable {
    iterable: f32,
    roots: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct DebugPlotDatum {
    x: f32,
    re: f32,
    im: f32,
    eps: f32,
}

trait RootsFinder {
    fn find_roots(&self, system: SystemType);
    fn debug(&self, system: SystemType);
}

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍  ", "");

struct FindRootsOptions {
    lower: f32,
    upper: f32,
    resolution: i32,
    patience: i32,
    tolerance: f32,
}

fn find_roots_using_newton<'a, F>(
    opts: FindRootsOptions,
    range_values: Vec<f32>,
    transcendental_equation: F,
) -> Vec<RootsPerIterable>
where
    F: Fn(Dual32, f32) -> Dual32 + Copy + Send + Sync + 'a,
{
    println!("{} Finding roots using Newton's method...", LOOKING_GLASS);
    let pb = ProgressBar::new(range_values.len() as u64);

    let mut roots: Vec<RootsPerIterable> = Vec::new();
    range_values
        .par_iter()
        .map(|k| {
            pb.inc(1);
            let root_equation = move |q: Dual32| transcendental_equation(q, *k);
            let result = root_search::<_, Dual32, f32>(
                root_equation,
                RootSearchOptions {
                    lower: opts.lower,
                    upper: opts.upper,
                    patience: opts.patience,
                    tolerance: opts.tolerance,
                    resolution: opts.resolution,
                },
            );
            RootsPerIterable {
                iterable: *k,
                roots: result.roots,
            }
        })
        .collect_into_vec(&mut roots);
    roots.sort_by(|a, b| a.iterable.partial_cmp(&b.iterable).unwrap());
    roots
}

fn find_roots_using_simple_search<'a, F>(
    opts: FindRootsOptions,
    range_values: Vec<f32>,
    transcendental_equation: F,
) -> Vec<RootsPerIterable>
where
    F: Fn(Dual32, f32) -> Dual32 + Copy + Send + Sync + 'a,
{
    println!("{} Finding roots using simple search...", LOOKING_GLASS);
    let pb = ProgressBar::new(range_values.len() as u64);

    let mut roots: Vec<RootsPerIterable> = Vec::new();
    range_values
        .par_iter()
        .map(|k| {
            pb.inc(1);
            let mut loop_values = Vec::new();
            for i in 0..opts.resolution {
                let step = (opts.upper - opts.lower) * i as f32 / (opts.resolution as f32);
                loop_values.push((
                    step,
                    transcendental_equation(Dual32::from_re(step), *k).re().re(),
                ));
            }
            RootsPerIterable {
                iterable: *k,
                roots: loop_values
                    .iter()
                    .filter(|e| e.1 == 0.0)
                    .map(|e| e.0 + opts.lower)
                    .collect(),
            }
        })
        .collect_into_vec(&mut roots);
    roots
}

impl RootsFinder for KronigPenney {
    fn find_roots(&self, system: SystemType) {
        let data: Vec<RootsPerIterable>;
        let opts = FindRootsOptions {
            lower: self.lower,
            upper: self.upper,
            resolution: self.resolution,
            patience: self.patience,
            tolerance: self.tolerance,
        };
        if self.lower < 0.0 && self.upper <= 0.0 {
            if self.u1 > 0.0 && self.u2 > 0.0 {
                panic!("No roots for negative energy and positive potential")
            }
            if system == SystemType::Bulk {
                let mut k_values: Vec<f32> = Vec::new();
                for i in 0..self.steps {
                    let step = 2.0 * i as f32 / (self.steps as f32);
                    let k = PI * (-1.0 + step);
                    k_values.push(k);
                }
                data = find_roots_using_newton(opts, k_values, |q: Dual32, k: f32| {
                    self.transcendental_equation_for_negative_energy_bulk_roots(q, k)
                });
            } else {
                let mut v_values: Vec<f32> = Vec::new();
                for i in 0..self.steps {
                    let step = self.d * i as f32 / (self.steps as f32);
                    let v = step;
                    v_values.push(v);
                }
                data = find_roots_using_simple_search(opts, v_values, |q: Dual32, v: f32| {
                    Dual32::from_re(
                        self.transcendental_equation_for_negative_energy_finite_roots(q, v)
                            .unwrap()
                            .re()
                            .re(),
                    )
                });
            };
        } else if self.lower >= 0.0 && self.upper > 0.0 {
            if self.u1 < 0.0 && self.u2 < 0.0 {
                panic!("No roots for positive energy and negative potential")
            }
            if system == SystemType::Bulk {
                let mut k_values: Vec<f32> = Vec::new();
                for i in 0..opts.resolution {
                    let step = 2.0 * i as f32 / (opts.resolution as f32);
                    let k = PI * (-1.0 + step);
                    k_values.push(k);
                }
                data = find_roots_using_newton(opts, k_values, |q: Dual32, k: f32| {
                    self.transcendental_equation_for_positive_energy_bulk_roots(q, k)
                })
            } else {
                panic!("Finite system not implemented yet for positive energy roots")
            }
        } else {
            panic!("Cannot find mixed roots, e.g. bound and propagating");
        }
        let file = std::fs::File::create(format!(
            "data/roots_system:{:?}_N:{}_u1:{}_u2:{}.json",
            system, self.n, self.u1, self.u2
        ))
        .unwrap();
        serde_json::to_writer_pretty(file, &data).unwrap();
    }

    fn debug(&self, system: SystemType) {
        let mut data: Vec<DebugPlotDatum> = Vec::new();
        if self.lower < 0.0 && self.upper <= 0.0 {
            if self.u1 > 0.0 && self.u2 > 0.0 {
                panic!("No roots for negative energy and positive potential")
            }
            let mut q_values: Vec<f32> = Vec::new();
            for i in 0..10000 {
                let step = ((self.upper - self.lower) * i as f32) / (10000 as f32);
                q_values.push(step);
            }
            if system == SystemType::Bulk {
                for q in q_values {
                    let val = self.transcendental_equation_for_negative_energy_bulk_roots(
                        Dual32::from_re(q),
                        1.0,
                    );
                    data.push(DebugPlotDatum {
                        x: q,
                        re: val.re().re(),
                        im: val.imaginary().re(),
                        eps: val.eps,
                    });
                }
            } else {
                for q in q_values {
                    let val = self.transcendental_equation_for_negative_energy_finite_roots(
                        Dual32::from_re(q),
                        self.v,
                    );
                    match val {
                        Some(val) => data.push(DebugPlotDatum {
                            x: q,
                            re: val.re().re(),
                            im: val.imaginary().re(),
                            eps: val.abs().eps,
                        }),
                        None => (),
                    }
                }
            };
        } else if self.lower >= 0.0 && self.upper > 0.0 {
            if self.u1 < 0.0 && self.u2 < 0.0 {
                panic!("No roots for positive energy and negative potential")
            }
            let mut q_values: Vec<f32> = Vec::new();
            for i in 0..10000 {
                let step = ((self.upper - self.lower) * i as f32) / (10000 as f32);
                q_values.push(step);
            }
            if system == SystemType::Bulk {
                for q in q_values {
                    let val = self.transcendental_equation_for_positive_energy_bulk_roots(
                        Dual32::from_re(q),
                        1.0,
                    );
                    data.push(DebugPlotDatum {
                        x: q,
                        re: val.re().re(),
                        im: val.imaginary().re(),
                        eps: val.abs().eps,
                    });
                }
            } else {
                panic!("Finite system not implemented yet for positive energy roots")
            }
        } else {
            panic!("Cannot handle mixed roots, e.g. bound and propagating");
        }
        let file = std::fs::File::create(format!(
            "data/debug__v:{}_system:{:?}_N:{}_u1:{}_u2:{}.json",
            self.v, system, self.n, self.u1, self.u2
        ))
        .unwrap();
        serde_json::to_writer_pretty(file, &data).unwrap();
    }
}

impl KronigPenney {
    fn new(
        u1: f32,
        u2: f32,
        d: f32,
        v: f32,
        n: u32,
        lower: f32,
        upper: f32,
        steps: i32,
        resolution: i32,
        patience: i32,
        tolerance: f32,
    ) -> Self {
        if lower > upper {
            panic!("Lower bound must be less than upper bound");
        }
        // let w = d - v;
        let x1 = -v / 2.0;
        let x2 = v / 2.0;
        Self {
            u1,
            u2,
            d,
            v,
            x1,
            x2,
            n,
            lower,
            upper,
            steps,
            resolution,
            patience,
            tolerance,
        }
    }

    fn transcendental_equation_for_positive_energy_bulk_roots(&self, q: Dual32, k: f32) -> Dual32 {
        let first_term = || -> Dual32 {
            let arg: Dual32 = k.into();
            arg.scale(self.d.into());
            let coeff = q.powi(2);
            coeff.scale(arg.cos());
            coeff
        };

        let second_term = || -> Dual32 {
            let diff = self.x2 - self.x1;
            diff.scale(2.0);

            let mut arg = q;
            let coeff1 = self.d - (2.0 * self.v);
            arg.scale(coeff1.into());

            let coeff2 = self.u1 * self.u2;
            arg = arg.cos();
            arg.scale(coeff2.into());

            arg
        };

        let third_term = || -> Dual32 {
            let coeff = q.powi(2) - Dual32::from_re((self.u1 * self.u2) as f32);
            let arg = q;
            arg.scale(self.d.into());
            coeff * arg.cos()
        };

        let fourth_term = || -> Dual32 {
            let coeff = q;
            coeff.scale((self.u1 + self.u2).into());
            let arg = q;
            arg.scale(self.d.into());
            coeff * arg.sin()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }

    fn transcendental_equation_for_negative_energy_bulk_roots(&self, q: Dual32, k: f32) -> Dual32 {
        let first_term = || -> Dual32 { q.powi(2) * (k * self.d).cos() };

        let second_term = || -> Dual32 { (q.powi(2) + (self.u1 * self.u2)) * (q * self.d).cosh() };

        let third_term = || -> Dual32 { q * (self.u1 + self.u2) * (q * self.d).sinh() };

        let fourth_term = || -> Dual32 {
            -Dual32::from_re(self.u1 * self.u2) * (q * (self.d - (2.0 * self.v))).cosh()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }

    fn x(&self, n: u32, v: f32) -> f32 {
        let length = match self.n % 2 {
            1 => 2.0 * self.d,
            0 => 2.0 * self.d + v,
            _ => panic!("Impossible"),
        };
        let x = match n % 2 {
            1 => v + ((n as i32 - 1) * self.d as i32 / 2) as f32,
            0 => n as f32 * self.d as f32 / 2.0,
            _ => panic!("Impossible"),
        };
        -length / 2.0 + x
    }

    fn transcendental_equation_for_negative_energy_finite_roots(
        &self,
        q: Dual32,
        v: f32,
    ) -> Option<Dual32> {
        let max = 2 * (self.n + 1) as usize;
        let determine_element = |i: usize, j: usize| {
            let arg = q;

            let u = match i % 2 {
                0 => self.u1,
                1 => self.u2,
                _ => panic!("Impossible"),
            };

            let exp_expression = |exp_sign: i8, j: usize| {
                let n = match j {
                    0 => 0,
                    m if m == max => self.n + 1,
                    _ => match j % 2 {
                        0 => u32::try_from(j / 2).unwrap(),
                        1 => u32::try_from((j + 1) / 2).unwrap(),
                        _ => panic!("Impossible"),
                    },
                };
                let coeff = (-exp_sign) as f32 * self.x(n, v);
                arg.scale(coeff.into());
                arg.exp()
            };

            let chi_expression =
                |chi_sign: i8| Dual32::from_re(u) - q.scale((chi_sign as f32).into());

            return match i % 2 {
                0 => {
                    if i == 0 {
                        if j == 0 {
                            exp_expression(1, j)
                        } else if j == 1 {
                            exp_expression(-1, j)
                        } else {
                            Dual32::from_re(0.0)
                        }
                    } else {
                        if j == i - 2 {
                            chi_expression(1) * exp_expression(1, j)
                        } else if j == i - 1 {
                            chi_expression(-1) * exp_expression(-1, j)
                        } else if j == i {
                            chi_expression(-1) * exp_expression(1, j)
                        } else if j == i + 1 {
                            chi_expression(1) * exp_expression(-1, j)
                        } else {
                            Dual32::from_re(0.0)
                        }
                    }
                }
                1 => {
                    if i == max {
                        if j == max {
                            exp_expression(-1, j)
                        } else if j == max - 1 {
                            exp_expression(1, j)
                        } else {
                            Dual32::from_re(0.0)
                        }
                    } else {
                        if j == i - 1 {
                            exp_expression(1, j)
                        } else if j == i {
                            exp_expression(-1, j)
                        } else if j == i + 1 {
                            exp_expression(1, j).scale(Dual::from_re(-1.0))
                        } else if j == i + 2 {
                            exp_expression(-1, j).scale(Dual::from_re(-1.0))
                        } else {
                            Dual32::from_re(0.0)
                        }
                    }
                }
                _ => {
                    panic!("Something went wrong in the calculation of the matrix.")
                }
            };
        };
        let mat = DMatrix::from_fn(max, max, determine_element);
        let lu = mat.full_piv_lu();
        Some(lu.determinant())
        // let svd = mat.try_svd_unordered(false, false, Dual32::from_re(1e-15), 100000);
        // match svd {
        //     Some(svd) => svd.singular_values.iter().copied().reduce(|acc, x| acc * x),
        //     None => None,
        // }
    }
}

fn main() {
    let kp = KronigPenney::new(
        -5.0, -5.0, 1.0, 0.2, 4, -50.0, 0.0, 100, 100000, 1000, 0.000001,
    );
    kp.find_roots(SystemType::Finite);
    // kp.debug(SystemType::Finite);
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_x() {
//         let kp = KronigPenney::new(
//             -5.0, -5.0, 1.0, 0.2, 4, -50.0, 0.0, 100, 1000000, 1000, 0.00001,
//         );
//         for n in 0..=kp.n + 1 {
//             println!("x({}) = {}", n, kp.x(n, 0.45));
//         }
//     }
// }
